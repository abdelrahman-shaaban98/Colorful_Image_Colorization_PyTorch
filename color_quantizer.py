import numpy as np
from sklearn.neighbors import NearestNeighbors


class ColorQuantizer:
    def __init__(self, ab_bins_path):
        self.ab_bins = np.load(ab_bins_path)  # (313, 2)
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(self.ab_bins)

    def encode(self, ab_image):
        # ab_image: (H, W, 2), numpy array of a,b channels
        h, w, _ = ab_image.shape
        ab_flat = ab_image.reshape(-1, 2)
        indices = self.nn.kneighbors(ab_flat, return_distance=False)
        return indices.reshape(h, w)

    def decode(self, prob_map):
        # prob_map: (313, H, W) numpy array of probabilities per bin
        ab = np.tensordot(prob_map, self.ab_bins, axes=([0], [0]))  # (H, W, 2)
        return ab.astype(np.float32)
    

if __name__ == "__main__":
    ab_bins_path = "./weights/pts_in_hull.npy" 
    quantizer = ColorQuantizer(ab_bins_path)