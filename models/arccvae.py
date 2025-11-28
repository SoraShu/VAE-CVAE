import torch
from torch import nn
from torch.nn import functional as F

class ArcCVAE(nn.Module):
    def __init__(self,
                 latent_dim: int = 512,
                 hidden_dims: list | None = None,
                 condition_grids: int = 7, # 6 train (in+out) + 1 test in
                 target_grids: int = 1,    # 1 test out
                 num_classes: int = 10,
                 img_size: int = 32,
                 **kwargs) -> None:
        super(ArcCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Channels configuration
        cond_channels = condition_grids * num_classes # 7 * 10 = 70
        target_channels = target_grids * num_classes  # 1 * 10 = 10
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # =====================================================
        # 1. Condition Encoder: extract condition features (P(c))
        # compress 70 channels context to feature embedding
        # =====================================================
        modules_cond = []
        c_in = cond_channels
        for h_dim in hidden_dims:
            modules_cond.append(
                nn.Sequential(
                    nn.Conv2d(c_in, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            c_in = h_dim
        self.condition_encoder = nn.Sequential(*modules_cond)
        
        self.enc_out_shape = self._get_conv_output_size(cond_channels, img_size, self.condition_encoder)
        self.flat_dim = self.enc_out_shape[0] * self.enc_out_shape[1] * self.enc_out_shape[2]
        
        self.fc_cond = nn.Linear(self.flat_dim, latent_dim)

        # =====================================================
        # 2. Posterior Encoder: q(z | x, c)
        # Input: Target (x) + Condition (c)
        # =====================================================
        modules_enc = []
        # Encoder receives concatenated x and c
        enc_in = target_channels + cond_channels 
        
        # To simplify, reuse a similar structure but with different channel numbers
        for h_dim in hidden_dims:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(enc_in, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            enc_in = h_dim
        self.posterior_encoder = nn.Sequential(*modules_enc)

        # Posterior outputs mu and logvar
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

        # =====================================================
        # 3. Decoder: p(x | z, c)
        # Input: Latent z + Encoded Condition
        # =====================================================
        modules_dec = []
        
        # Input dimension = z (latent_dim) + condition_embedding (latent_dim)
        self.decoder_input = nn.Linear(latent_dim * 2, self.flat_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules_dec)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            # Output: 1 grid (10 classes)
                            nn.Conv2d(hidden_dims[-1], out_channels=num_classes,
                                      kernel_size=3, padding=1),
                            # Output Logits
                            )

    def _get_conv_output_size(self, in_channels, img_size, model):
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            output = model(dummy_input)
        return output.shape[1:]

    def _to_one_hot(self, tensor, channels_dim=True):
        """Helper: LongTensor (B, C, H, W) -> FloatTensor (B, C*10, H, W)"""
        B, G, H, W = tensor.shape
        x_onehot = F.one_hot(tensor.long(), num_classes=self.num_classes).float()
        # (B, G, H, W, 10) -> (B, G, 10, H, W) -> (B, G*10, H, W)
        return x_onehot.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)

    def encode_condition(self, condition):
        """Maps condition grids to a latent embedding"""
        # condition: (B, 7, 32, 32)
        c = self._to_one_hot(condition) # (B, 70, 32, 32)
        c_feat = self.condition_encoder(c) # (B, 512, 2, 2)
        c_flat = torch.flatten(c_feat, start_dim=1)
        c_emb = self.fc_cond(c_flat) # (B, latent_dim)
        return c_emb

    def encode_posterior(self, target, condition):
        """Maps (target, condition) to mu, logvar"""
        # target: (B, 1, 32, 32), condition: (B, 7, 32, 32)
        t = self._to_one_hot(target) # (B, 10, 32, 32)
        c = self._to_one_hot(condition) # (B, 70, 32, 32)
        
        x = torch.cat([t, c], dim=1) # (B, 80, 32, 32)
        
        res = self.posterior_encoder(x)
        res = torch.flatten(res, start_dim=1)
        
        mu = self.fc_mu(res)
        log_var = self.fc_var(res)
        return mu, log_var

    def decode(self, z, c_emb):
        # z: (B, latent_dim), c_emb: (B, latent_dim)
        # Concatenate z and condition embedding
        z_c = torch.cat([z, c_emb], dim=1) 
        
        result = self.decoder_input(z_c)
        result = result.view(-1, *self.enc_out_shape)
        result = self.decoder(result)
        result = self.final_layer(result) # (B, 10, 32, 32)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, target, condition, **kwargs):
        # 1. Encode Condition
        c_emb = self.encode_condition(condition)
        
        # 2. Encode Posterior (Training only, needs target)
        mu, log_var = self.encode_posterior(target, condition)
        
        # 3. Sample z
        z = self.reparameterize(mu, log_var)
        
        # 4. Decode (z + condition)
        recons_logits = self.decode(z, c_emb)
        
        return [recons_logits, target, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons_logits = args[0]
        target = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']

        # Cross Entropy
        # recons: (B, 10, 32, 32), target: (B, 1, 32, 32) -> squeeze to (B, 32, 32)
        target_flat = target.squeeze(1).long()
        ce_loss = F.cross_entropy(recons_logits, target_flat)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = ce_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': ce_loss, 'KLD': -kld_loss}
    
    def predict(self, condition):
        """Inference time: generate prediction given only condition"""
        # 1. Encode Condition
        c_emb = self.encode_condition(condition)
        
        # 2. Sample z from Prior N(0, I)
        z = torch.randn(condition.size(0), self.latent_dim).to(condition.device)
        
        # 3. Decode
        logits = self.decode(z, c_emb)
        pred = torch.argmax(logits, dim=1) # (B, 32, 32)
        return pred