import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
from models import ArcCVAE

# ==========================================
# 1. CVAE Dataset
# ==========================================
class ARCCVAEDataset(Dataset):
    def __init__(self, data_dir, mode='training', img_size=32):
        self.data_path = os.path.join(data_dir, mode)
        self.file_list = glob.glob(os.path.join(self.data_path, '*.json'))
        self.data_points = []
        self.img_size = img_size
        
        print(f"Processing {mode} data (CVAE)...")
        self._process_files()
        print(f"Found {len(self.data_points)} valid samples.")

    def _pad_grid(self, grid):
        grid = np.array(grid)
        h, w = grid.shape
        pad_h = max(0, self.img_size - h)
        pad_w = max(0, self.img_size - w)
        if h > self.img_size or w > self.img_size:
             grid = grid[:self.img_size, :self.img_size]
             pad_h = max(0, self.img_size - grid.shape[0])
             pad_w = max(0, self.img_size - grid.shape[1])
        return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    def _process_files(self):
        for fpath in tqdm(self.file_list): # pyrefly: ignore[not-iterable]
            try:
                with open(fpath, 'r') as f:
                    task = json.load(f)
                
                train_pairs = task['train']
                test_pairs = task['test']

                if len(train_pairs) != 3:
                    continue

                # --- Condition Part 1: Train Pairs (6 grids) ---
                train_context = []
                for pair in train_pairs:
                    train_context.append(self._pad_grid(pair['input']))
                    train_context.append(self._pad_grid(pair['output']))
                
                # Iterate over test pairs
                for t_pair in test_pairs:
                    # --- Condition Part 2: Test Input (1 grid) ---
                    test_input = self._pad_grid(t_pair['input'])
                    
                    # Full Condition: Train Pairs + Test Input (7 grids)
                    condition = np.array(train_context + [test_input]) 
                    
                    # --- Target: Test Output (1 grid) ---
                    target = self._pad_grid(t_pair['output'])
                    # Add channel dimension to target: (1, 32, 32)
                    target = np.expand_dims(target, axis=0)
                    
                    self.data_points.append((condition, target))
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        cond_arr, target_arr = self.data_points[index]
        return torch.from_numpy(cond_arr).long(), torch.from_numpy(target_arr).long()

# ==========================================
# 2. Training Loops
# ==========================================

def train_one_epoch(model, loader, optimizer, device, kld_weight):
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for condition, target in pbar: #pyrefly: ignore[not-iterable]
        # condition: (B, 7, 32, 32)
        # target: (B, 1, 32, 32)
        condition, target = condition.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # CVAE Forward: needs both target and condition
        results = model(target, condition) 
        
        loss_dict = model.loss_function(*results, M_N=kld_weight)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy check
        logits = results[0]
        preds = torch.argmax(logits, dim=1)
        # target is (B, 1, 32, 32), preds is (B, 32, 32)
        acc = (preds == target.squeeze(1)).float().mean()
        total_acc += acc.item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc.item():.4f}"}) #pyrefly: ignore[missing-attribute]
        
    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_acc = 0
    total_asis_acc = 0
    total_exact_match = 0
    
    with torch.no_grad():
        for condition, target in loader:
            condition, target = condition.to(device), target.to(device)
            

            preds = model.predict(condition)
            
            target_sq = target.squeeze(1)
            acc = (preds == target_sq).float().mean()
            # as-is acc: compare input and target
            asis_acc = (condition[:, -1, :, :] == target_sq).float().mean()
            total_acc += acc.item()
            total_asis_acc += asis_acc.item()
            
            matches = (preds == target_sq).view(preds.size(0), -1).all(dim=1).float().mean()
            total_exact_match += matches.item()
            
    return total_acc / len(loader), total_exact_match / len(loader), total_asis_acc / len(loader)

def visualize_results(model, loader, device, epoch):
    model.eval()
    try:
        condition, target = next(iter(loader))
    except StopIteration:
        return

    condition, target = condition.to(device), target.to(device)
    
    with torch.no_grad():
        preds = model.predict(condition) # (B, 32, 32)
    
    num = min(condition.size(0), 8)
    
    # Extract Test Input from Condition (Index 6)
    test_input = condition[:num, 6, :, :].cpu().float()
    target_img = target[:num, 0, :, :].cpu().float()
    pred_img = preds[:num, :, :].cpu().float()
    
    # Normalize
    test_input /= 9.0
    target_img /= 9.0
    pred_img /= 9.0
    
    # (B, 1, H, W)
    comparison = torch.cat([test_input.unsqueeze(1), target_img.unsqueeze(1), pred_img.unsqueeze(1)], dim=3)
    
    os.makedirs("results_arc_cvae", exist_ok=True)
    save_image(comparison, f"results_arc_cvae/epoch_{epoch}.png")

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/arc2', help='Root dir')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=256)
    args = parser.parse_args()

    IMG_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = ARCCVAEDataset(args.data_dir, mode='training', img_size=IMG_SIZE)
    val_dataset = ARCCVAEDataset(args.data_dir, mode='evaluation', img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # CVAE: condition=7 grids, target=1 grid
    model = ArcCVAE(latent_dim=args.latent_dim, 
                    condition_grids=7, 
                    target_grids=1, 
                    num_classes=10, 
                    img_size=IMG_SIZE).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_avg_acc = 0.0
    for epoch in range(args.epochs):
        # KLD Annealing
        kld_weight = min(0.01, (epoch / 20) * 0.01)
        
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, device, kld_weight)
        val_acc, val_exact,_ = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f} | Val Exact: {val_exact:.4f}")
        
        if val_acc > best_avg_acc:
            best_avg_acc = val_acc
            torch.save(model.state_dict(), "results_arc_cvae/best_cvae.pth")

        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device, epoch+1)
            torch.save(model.state_dict(), f"results_arc_cvae/cvae_{epoch+1}.pth")
        
        # load best model for final evaluation
        model.load_state_dict(torch.load("results_arc_cvae/best_cvae.pth"))
        val_acc, val_exact, val_asis = evaluate(model, val_loader, device)
        print(f"Best Model | Val Acc: {val_acc:.4f} | Val Exact: {val_exact:.4f} | Val As-Is: {val_asis:.4f}")
        visualize_results(model, val_loader, device, "best")

if __name__ == "__main__":
    main()