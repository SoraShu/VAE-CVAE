import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm


from models import VAE, CVAE


# config
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
LATENT_DIM = 8
HIDDEN_DIMS = [128]
KLD_WEIGHT = 0.00025

def train_model(model_type:str, train_loader, device, epochs, lr, latent_dim, hidden_dims, kld_weight):
    # 1. initialize model
    match model_type:
        case 'vae':
            model = VAE(in_channels=1, latent_dim=latent_dim, hidden_dims=hidden_dims, img_size=28).to(device)
        case 'cvae':
            model = CVAE(in_channels=1, num_classes=10, latent_dim=latent_dim, hidden_dims=hidden_dims, img_size=28).to(device)
        case _:
            raise ValueError("Invalid model type. Choose 'vae' or 'cvae'.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, labels) in enumerate(pbar): # pyrefly: ignore[bad-argument-type,bad-assignment]
            data = data.to(device)
            
            # cvae needs one-hot labels
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            
            optimizer.zero_grad()
            match model_type:
                case 'vae':
                    results = model(data)
                    loss_dict = model.loss_function(*results, M_N=kld_weight)
                case 'cvae':
                    # CVAE forward needs labels
                    results = model(data, labels=labels)
                    loss_dict = model.loss_function(*results, M_N=kld_weight)
                case _:
                    raise ValueError()
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss = loss_dict['Reconstruction_Loss'].item()
            kld_loss_val = loss_dict['KLD'].item()
            
            total_recon += recon_loss
            total_kld += kld_loss_val
            
            pbar.set_postfix({ # pyrefly: ignore[missing-attribute]
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss:.4f}',
                'KLD': f'{kld_loss_val:.4f}'
            })

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | Recon: {total_recon / len(train_loader.dataset):.6f} | KLD: {total_kld / len(train_loader.dataset):.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'results_mnist/best_{model_type}_mnist.pth')
    
    # 2. Sample generation after training
    print("Generating samples...")
    model.eval()
    with torch.no_grad():
        # Generate 64 samples (8x8 Grid)
        z = torch.randn(64, latent_dim).to(device)

        match model_type:
            case 'vae':
                sample = model.decode(z).cpu()
            case 'cvae':
                # CVAE: generate 8 samples for each digit 0-7
                # Construct labels: 00000000 11111111 ...
                digit_labels = torch.arange(0, 8).repeat_interleave(8).to(device)
                labels_one_hot = torch.nn.functional.one_hot(digit_labels, num_classes=10).float().to(device)
                z_input = torch.cat((z, labels_one_hot), dim=1)
                sample = model.decode(z_input).cpu()
            case _:
                raise ValueError()

        save_image(sample.view(64, 1, 28, 28),
                   f'results_mnist/{model_type}_sample_epoch_{epochs}.png')
    
    print(f'Model {model_type} training complete. Sample images saved.')

# load best model and evaluate on test set
def evaluate_model(model_type:str, test_loader, device, latent_dim, hidden_dims):
    match model_type:
        case 'vae':
            model = VAE(in_channels=1, latent_dim=latent_dim, hidden_dims=hidden_dims, img_size=28).to(device)
        case 'cvae':
            model = CVAE(in_channels=1, num_classes=10, latent_dim=latent_dim, hidden_dims=hidden_dims, img_size=28).to(device)
        case _:
            raise ValueError("Invalid model type. Choose 'vae' or 'cvae'.")
    model_path = f'results_mnist/best_{model_type}_mnist.pth'
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}, skipping evaluation.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader): # pyrefly: ignore[bad-argument-type,bad-assignment]
            data = data.to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            
            if model_type == 'vae':
                results = model(data)
                loss_dict = model.loss_function(*results, M_N=1.0)
            else:
                results = model(data, labels=labels)
                loss_dict = model.loss_function(*results, M_N=1.0)
                
            total_loss += loss_dict['loss'].item()
    
    print(f"Test Set Average Loss ({model_type}): {total_loss / len(test_loader.dataset):.6f}")

    # sample generation
    print("Generating samples...")
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)

        match model_type:
            case 'vae':
                sample = model.decode(z).cpu()
            case 'cvae':
                digit_labels = torch.arange(0, 8).repeat_interleave(8).to(device)
                labels_one_hot = torch.nn.functional.one_hot(digit_labels, num_classes=10).float().to(device)
                z_input = torch.cat((z, labels_one_hot), dim=1)
                sample = model.decode(z_input).cpu()
            case _:
                raise ValueError()

        save_image(sample.view(64, 1, 28, 28),
                   f'results_mnist/{model_type}_test_sample.png')



def main():
    # parse arguments
    argparser = argparse.ArgumentParser(description='VAE/CVAE MNIST Trainer')
    argparser.add_argument('--model', type=str, choices=['vae', 'cvae'], default='vae', help='Model type to train (vae or cvae)')
    argparser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    argparser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    argparser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    argparser.add_argument('--latent-dim', type=int, default=LATENT_DIM, help='Dimensionality of the latent space')
    # nargs='+' allows passing a list, e.g., --hidden-dims 32 64 128
    argparser.add_argument('--hidden-dims', type=int, nargs='+', default=HIDDEN_DIMS, help='List of hidden layer dimensions')
    argparser.add_argument('--kld-weight', type=float, default=KLD_WEIGHT, help='Weight for the KLD loss term')
    args = argparser.parse_args()

    os.makedirs('results_mnist', exist_ok=True)
    

    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(args.model, train_loader, device, args.epochs, args.lr, args.latent_dim, args.hidden_dims, args.kld_weight)
    evaluate_model(args.model, test_loader, device, args.latent_dim, args.hidden_dims)

if __name__ == "__main__":
    main()