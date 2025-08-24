import torch
import torch.nn.functional as F
import cv2
import numpy as np
from guided_filter_pytorch.guided_filter import GuidedFilter
import scipy.sparse
import scipy.sparse.linalg
from skimage import feature


def compute_edge_map(gray_tensor, device, sigma=1.5):
    """
    Generate an edge map (batch supported).
    """
    B = gray_tensor.shape[0]
    edge_maps = []

    for b in range(B):
        gray_np = gray_tensor[b, 0].detach().cpu().numpy()  # (H, W)
        gray_np = np.clip(gray_np, 0, 1).astype(np.float32)
        edge = feature.canny(gray_np, sigma=float(sigma))  # (H, W)
        edge_tensor = torch.tensor(edge, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        edge_maps.append(edge_tensor)

    return torch.cat(edge_maps, dim=0).to(device)  # (B, 1, H, W)

def estimate_sparse_blur(gray_image, edge_map, device):
    """
    Generate a sparse blur map (operate at 299x299 from the start).
    """
    target_size = (299, 299)  
    
    gray_image = F.interpolate(gray_image, size=target_size, mode="bilinear", align_corners=False)
    edge_map = F.interpolate(edge_map, size=target_size, mode="bilinear", align_corners=False)

    std1, std2 = 1.5, 2.0
    # std1, std2 = 2.0, 2.5
    
    def gaussian_gradient(size, sigma, device):
        coords = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(coords, coords, indexing="ij")
        g_x = -x / (2 * torch.pi * sigma**4) * torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        g_y = -y / (2 * torch.pi * sigma**4) * torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g_x.unsqueeze(0).unsqueeze(0), g_y.unsqueeze(0).unsqueeze(0)

    g1x, g1y = gaussian_gradient(11, std1, device)
    g2x, g2y = gaussian_gradient(11, std2, device)

    gimx1 = F.conv2d(gray_image, g1x, padding=5)
    gimy1 = F.conv2d(gray_image, g1y, padding=5)
    gimx2 = F.conv2d(gray_image, g2x, padding=5)
    gimy2 = F.conv2d(gray_image, g2y, padding=5)

    mg1 = torch.sqrt(gimx1**2 + gimy1**2)
    mg2 = torch.sqrt(gimx2**2 + gimy2**2)

    R = mg1 / (mg2 + 1e-8)
    edge_weight = edge_map / (edge_map.max() + 1e-8)
    R = R * edge_weight  

    sparse_vals = (R**2 * std1**2 - std2**2) / (1 - R**2 + 1e-8)
    sparse_vals = torch.clamp(sparse_vals, min=0)
    sparse_bmap = torch.sqrt(sparse_vals)

    sparse_bmap = torch.nan_to_num(sparse_bmap, nan=0.0)
    sparse_bmap = torch.clamp(sparse_bmap, max=5)

    return sparse_bmap  

def closed_form_matting(image, sparse_bmap, device):
    """
    Apply closed-form matting to produce the final defocus map
    """
    target_size = (299, 299)  
    image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)
    sparse_bmap = F.interpolate(sparse_bmap, size=target_size, mode="bilinear", align_corners=False)

    def final_defocus_map(image_tensor, sparse_bmap, radius=7, eps=1e-2):
        guide = torch.mean(image_tensor, dim=1, keepdim=True)
        gf = GuidedFilter(radius, eps)
        refined_map = gf(guide, sparse_bmap)
        return refined_map

    final_bmap = final_defocus_map(image, sparse_bmap)

    return final_bmap

