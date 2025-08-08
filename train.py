import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from color_quantizer import ColorQuantizer
from model import ColorizationNet
from dataset import ColorizationDataset
from utils import RebalancedCrossEntropy

# print(torch.cuda.is_available())         
# print(torch.cuda.get_device_name(0))     


def train(model, dataloader, optimizer, criterion, scaler, save_dir="./weights/model"):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for l_image, label in tqdm(dataloader):
            l_image = l_image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(l_image)
                loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * l_image.size(0)

        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, f"epoch_{epoch + 1}_loss_{best_loss}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to: {save_path}")
            

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l_channel_path = "./data/l_channel"
    ab_channels_path = "./data/ab_channels"
    ab_bins_path = "./weights/pts_in_hull.npy"
    classes_weights_path = "./weights/class_weights.npy"

    model = ColorizationNet().to(device)
    quantizer = ColorQuantizer(ab_bins_path)
    dataset = ColorizationDataset(l_channel_path, ab_channels_path, quantizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = RebalancedCrossEntropy(classes_weights_path, device)
    scaler = GradScaler()

    train(model, dataloader, optimizer, criterion, scaler)