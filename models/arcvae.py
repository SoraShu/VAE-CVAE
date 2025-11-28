import torch
from torch import nn
from torch.nn import functional as F

class ArcVAE(nn.Module):
    def __init__(self,
                 latent_dim: int = 512,
                 hidden_dims: list| None = None,
                 input_grids: int = 7, # 3 train pairs (in+out) + 1 test input
                 num_classes: int = 10, # 0-9 colors
                 img_size: int = 32,
                 **kwargs) -> None:
        super(ArcVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.input_grids = input_grids

        # input_channels = grids * classes (one-hot expanded)
        # 7 * 10 = 70 channels
        in_channels = input_grids * num_classes
        
        modules = []
        if hidden_dims is None:
            # 32x32 -> 16x16 -> 8x8 -> 4x4
            hidden_dims = [128, 256, 512]

        # Build Encoder
        current_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate flattened size dynamically
        self.enc_out_shape = self._get_encoder_output_size(in_channels, img_size)
        self.enc_out_flat_dim = self.enc_out_shape[0] * self.enc_out_shape[1] * self.enc_out_shape[2]

        self.fc_mu = nn.Linear(self.enc_out_flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.enc_out_flat_dim, latent_dim)

        # Build Decoder
        # Decoder needs to reconstruct 1 grid with 10 classes
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.enc_out_flat_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1), # cautious with padding on 32x32
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            # Output: 1 grid * 10 classes
                            nn.Conv2d(hidden_dims[-1], out_channels=num_classes,
                                      kernel_size=3, padding=1),
                            # No activation here, we output Logits for CrossEntropy
                            )

    def _get_encoder_output_size(self, in_channels, img_size):
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            output = self.encoder(dummy_input)
        return output.shape[1:]

    def encode(self, input):
        """
        input: [Batch, 7, 32, 32] (Integer indices)
        We need to convert to One-Hot: [Batch, 70, 32, 32]
        """
        # One Hot Encoding inside the model to save memory in Dataloader
        # Input shape: (B, 7, 32, 32)
        B, G, H, W = input.shape
        
        # (B, G, H, W) -> (B, G, H, W, 10)
        x_onehot = F.one_hot(input.long(), num_classes=self.num_classes).float()
        
        # Permute to (B, G, 10, H, W) -> Flatten G and 10 -> (B, 70, H, W)
        x = x_onehot.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, *self.enc_out_shape)
        result = self.decoder(result)
        result = self.final_layer(result) # Output: (B, 10, 30, 30) Logits
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons_logits = args[0] # (B, 10, 32, 32)
        # input = args[1] # We don't use input for reconstruction, we use target
        target = kwargs['target'] # (B, 32, 32) indices
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']

        # Cross Entropy Loss for classification
        # recons_logits: (B, 10, 32, 32), target: (B, 32, 32)
        ce_loss = F.cross_entropy(recons_logits, target.long())

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = ce_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': ce_loss, 'KLD': -kld_loss}
    
    def predict(self, input):
        """Returns the integer grid prediction"""
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var) # Or just use mu for deterministic
        logits = self.decode(z) # (B, 10, 32, 32)
        pred = torch.argmax(logits, dim=1) # (B, 32, 32)
        return pred