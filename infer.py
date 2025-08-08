import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import cv2
import torch
from model import ColorizationNet
from color_quantizer import ColorQuantizer
from utils import annealed_mean_softmax


def colorize_image(rgb_image, model, quantizer, device, T=0.5, img_size=(256, 256)):
    model.eval()
    
    with torch.no_grad():
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0  

        lab = rgb2lab(rgb_image)
        L = lab[..., 0:1] / 100.0  
        orig_size = L.shape[:2]

        L_resized = cv2.resize(L, img_size)
        L_tensor = torch.from_numpy(L_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        logits = model(L_tensor)
        probs = annealed_mean_softmax(logits, T=T)[0].cpu().numpy()  # (313, H', W')
        ab_pred = quantizer.decode(probs)  # (H', W', 2)

        ab_upsampled = cv2.resize(ab_pred, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)

        L_orig = L * 100 
        
        l = np.concatenate([L_orig, ab_upsampled], axis=-1)
        rgb_out = lab2rgb(l)
        
        return L_orig, rgb_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationNet().to(device)
    model.load_state_dict(torch.load('./weights/model/ColorizationNetWeights.pt', map_location=device, weights_only=True))
    ab_bins_path = "./weights/pts_in_hull.npy"
    quantizer = ColorQuantizer(ab_bins_path)

    img = Image.open("./examples/gray_sample.jpg")
    img_np = np.array(img) 
    L_orig, img_out = colorize_image(img_np, model, quantizer, device)