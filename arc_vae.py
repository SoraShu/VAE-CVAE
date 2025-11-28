import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
from models import ArcVAE

# ==========================================
# 1. Dataset Definition
# ==========================================
class ARCDataset(Dataset):
    def __init__(self, data_dir, mode='training', img_size=32):
        self.data_path = os.path.join(data_dir, mode)
        self.file_list = glob.glob(os.path.join(self.data_path, '*.json'))
        self.data_points = [] 
        self.img_size = img_size 
        
        print(f"Processing {mode} data from {self.data_path}...")
        self._process_files()
        print(f"Found {len(self.data_points)} valid samples in {mode}.")

    def _pad_grid(self, grid):
        """Pads a grid to img_size x img_size with 0s."""
        grid = np.array(grid)
        h, w = grid.shape
        pad_h = max(0, self.img_size - h)
        pad_w = max(0, self.img_size - w)
        
        # crop if larger than img_size
        if h > self.img_size or w > self.img_size:
             grid = grid[:self.img_size, :self.img_size]
             # recalculate padding
             pad_h = max(0, self.img_size - grid.shape[0])
             pad_w = max(0, self.img_size - grid.shape[1])

        # Pad with 0 (background color)
        # mode='constant', constant_values=0
        return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    def _process_files(self):
        for fpath in tqdm(self.file_list):# pyrefly: ignore[not-iterable]
            try:
                with open(fpath, 'r') as f:
                    task = json.load(f)
                
                train_pairs = task['train']
                test_pairs = task['test']

                # Constraint: Exactly 3 train pairs
                if len(train_pairs) != 3:
                    continue

                # Prepare Training Context
                context_grids = []
                for pair in train_pairs:
                    context_grids.append(self._pad_grid(pair['input']))
                    context_grids.append(self._pad_grid(pair['output']))
                
                # Iterate over test pairs
                for t_pair in test_pairs:
                    test_input = self._pad_grid(t_pair['input'])
                    full_input = np.array(context_grids + [test_input]) 
                    target_output = self._pad_grid(t_pair['output']) 
                    
                    self.data_points.append((full_input, target_output))
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        input_arr, target_arr = self.data_points[index]
        return torch.from_numpy(input_arr).long(), torch.from_numpy(target_arr).long()

# ==========================================
# 2. Training & Evaluation Logic
# ==========================================

def train_one_epoch(model, loader, optimizer, device, kld_weight):
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar: # pyrefly: ignore[not-iterable]
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        results = model(inputs) 
        
        loss_dict = model.loss_function(*results, target=targets, M_N=kld_weight)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        logits = results[0]
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        total_acc += acc.item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc.item():.4f}"})# pyrefly: ignore[missing-attribute]
        
    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_acc = 0
    total_full_grid_acc = 0 
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model.predict(inputs) 
            
            acc = (preds == targets).float().mean()
            total_acc += acc.item()
            
            matches = (preds == targets).view(preds.size(0), -1).all(dim=1).float().mean()
            total_full_grid_acc += matches.item()
            
    return total_acc / len(loader), total_full_grid_acc / len(loader)

def visualize_results(model, loader, device, epoch):
    model.eval()
    # Handle case where batch is smaller than 8
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        return

    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model.predict(inputs) 
    
    # Take up to 8 samples
    num_samples = min(inputs.size(0), 8)
    
    test_inputs = inputs[:num_samples, 6, :, :].cpu().float()
    targets = targets[:num_samples, :, :].cpu().float()
    preds = preds[:num_samples, :, :].cpu().float()
    
    test_inputs /= 9.0
    targets /= 9.0
    preds /= 9.0
    
    comparison = torch.cat([test_inputs.unsqueeze(1), targets.unsqueeze(1), preds.unsqueeze(1)], dim=3)
    
    os.makedirs("results_arc", exist_ok=True)
    save_image(comparison, f"results_arc/epoch_{epoch}.png")

# ==========================================
# 3. Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/arc2', help='Root dir containing training/ and evaluation/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=256)
    args = parser.parse_args()

    # pad to 32x32
    IMG_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    train_dataset = ARCDataset(args.data_dir, mode='training', img_size=IMG_SIZE)
    val_dataset = ARCDataset(args.data_dir, mode='evaluation', img_size=IMG_SIZE)
    
    # drop_last=True is important for BatchNorm
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # For validation, if dataset is small, drop_last might lose too much data, but it's safer for dimension errors
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. Initialize Model
    model = ArcVAE(latent_dim=args.latent_dim, 
                   input_grids=7, 
                   num_classes=10, 
                   img_size=IMG_SIZE).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    kld_weight = 0.0

    best_train_acc = 0.0
    
    for epoch in range(args.epochs):
        kld_weight = min(0.01, (epoch / 20) * 0.01)
        
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, device, kld_weight)
        val_acc, val_exact_match = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f} | Val Exact: {val_exact_match:.4f}")
        
        if avg_acc > best_train_acc:
            best_train_acc = avg_acc
            torch.save(model.state_dict(), "results_arc/arc_vae_best.pth")
        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device, epoch+1)
            torch.save(model.state_dict(), f"results_arc/arc_vae_{epoch+1}.pth")
    
    # load best model for final evaluation
    model.load_state_dict(torch.load("results_arc/arc_vae_best.pth"))
    val_acc, val_exact_match = evaluate(model, val_loader, device)
    print(f"Final Evaluation on Validation Set | Val Acc: {val_acc:.4f} | Val Exact: {val_exact_match:.4f}")
    visualize_results(model, val_loader, device, 'final')

if __name__ == "__main__":
    main()