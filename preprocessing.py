import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.color import rgb2lab
from tqdm import tqdm
from color_quantizer import ColorQuantizer 


def compute_class_weights(image_paths, quantizer, img_size=(256, 256), lambda_val=0.5):
    counts = np.zeros(313, dtype=np.float64)

    for path in tqdm(image_paths, desc="Computing color bin frequencies"):
        img_in = Image.open(path).resize(img_size)
        img_out = img_in.resize((img_size[0] // 4, img_size[1] // 4))
        lab_out = rgb2lab(img_out)
        ab_out = lab_out[..., 1:]

        ab_out = ab_out.reshape(-1, 2)
        labels = quantizer.nn.kneighbors(ab_out, return_distance=False).flatten()

        for idx in labels:
            counts[idx] += 1

    # Convert to smoothed inverse freq weights
    p = counts / counts.sum()  # empirical probabilities
    Q = len(p)
    rebalance = 1.0 / ((1 - lambda_val) * p + lambda_val * (1 / Q))
    rebalance = rebalance / rebalance.mean()  # normalize to mean 1
    np.save("./weights/class_weights.npy", rebalance)
    return rebalance


def generate_training_data(image_paths, l_channel_path="./data/l_channel", ab_channels_small_path="./data/ab_channels"):
    # Directories to save output
    os.makedirs(l_channel_path, exist_ok=True)
    os.makedirs(ab_channels_small_path, exist_ok=True)

    # Resize and transform pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # Converts to [0,1] range, shape (C, H, W)
    ])

    for img_path in tqdm(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)  # (3, 256, 256)

        # Convert to numpy for LAB
        img_np = img_tensor.permute(1, 2, 0).numpy()  # (256, 256, 3)
        lab = rgb2lab(img_np).astype(np.float32)  # LAB image
        
        l = lab[:, :, 0:1]      # (256, 256, 1)
        ab = lab[:, :, 1:]      # (256, 256, 2)
        
        # Transpose to (C, H, W)
        l_tensor = torch.from_numpy(l).permute(2, 0, 1)      # (1, 256, 256)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)    # (2, 256, 256)
        ab_tensor_small = F.interpolate(ab_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0) # (2, 64, 64)

        torch.save(l_tensor, f'{l_channel_path}/{img_name}_l.pt')
        torch.save(ab_tensor_small, f'{ab_channels_small_path}/{img_name}_ab_small.pt')
        

if __name__ == "__main__":
    image_paths = "./data/Flickr_8k/Images" 
    image_paths = glob.glob(image_paths + "/*.jpg") 
    ab_bins_path = "./weights/pts_in_hull.npy"
    quantizer = ColorQuantizer(ab_bins_path)

    compute_class_weights(image_paths, quantizer)
    generate_training_data(image_paths)