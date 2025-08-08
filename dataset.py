import glob
import torch
from torch.utils.data import Dataset, DataLoader


class ColorizationDataset(Dataset):
    def __init__(self, l_channel_path, ab_channels_path, quantizer):
        self.l_channel_path = glob.glob(l_channel_path + "/*.pt") 
        self.ab_channels_path = glob.glob(ab_channels_path + "/*.pt") 
        self.quantizer = quantizer
        
    def __len__(self):
        return len(self.l_channel_path)

    def __getitem__(self, idx):
        l_in = torch.load(self.l_channel_path[idx], weights_only=True) / 100.0  # normalize L to [0,1]
        ab_out = torch.load(self.ab_channels_path[idx], weights_only=True)
        ab_out = ab_out.permute(1, 2, 0).numpy()
        label = self.quantizer.encode(ab_out)  # (H/4, W/4)
 
        return l_in.float(), torch.from_numpy(label).long()        
        

if __name__ == "__main__":
    from color_quantizer import ColorQuantizer
    quantizer = ColorQuantizer("./weights/pts_in_hull.npy")
    dataset = ColorizationDataset("./data/l_channel", "./data/ab_channels", quantizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    for l, ab in dataloader:
        print(l.max())
        print(l.shape)
        print(ab.max())
        print(ab.shape)
        break