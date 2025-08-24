import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from util.defocus_processing import compute_edge_map, estimate_sparse_blur, closed_form_matting
import timm  
from util.timer import StepTimer
import torch
import torch.distributed as dist
import csv



def run_defocus_pipeline(image, device):
    timer = StepTimer(device)

    gray_image = timer.timeit(
        "rgb_to_gray",
        torch.mean, image,  dim=1, keepdim=True
    )

    edge_map = timer.timeit(
        "edge_map",
        compute_edge_map, gray_image, device
    )

    sparse_bmap = timer.timeit(
        "sparse_blur",
        estimate_sparse_blur, gray_image, edge_map, device
    )

    blended_tensor = timer.timeit(
        "closed_form_matting",
        closed_form_matting, image, sparse_bmap, device
    )

    metrics = timer.summary()   # per-stage time in ms and peak memory in bytes
    return blended_tensor, sparse_bmap, edge_map, metrics

# Physics-based Defocus Map generator Module
class DefocusMapGenerator(nn.Module):
    def __init__(self):
        super(DefocusMapGenerator, self).__init__()

    def forward(self, image, device):
        """
        - image: RGB tensor of shape (B, 3, H, W), normalized to 0..1
        - device: execution device ("cuda" or "cpu")
        - returns: defocus map (B, 1, H, W) and intermediates with timing
        """
        return run_defocus_pipeline(image, device)  # measure per-stage performance with the timer
        # # RGB → Grayscale
        # gray_image = torch.mean(image, dim=1, keepdim=True)  # (B, 1, H, W)

        # # Edge Map
        # edge_map = compute_edge_map(gray_image, device)  # (B, 1, H, W)

        # # Sparse Blur Map
        # sparse_bmap = estimate_sparse_blur(gray_image, edge_map, device)  # (B, 1, H, W)

        # # Closed-Form Matting 
        # blended_tensor = closed_form_matting(image, sparse_bmap, device)  # (B, 1, H, W)

        # return blended_tensor, sparse_bmap, edge_map  # return 3 maps
        
class DefocusNet(nn.Module):
    def __init__(self, num_classes, backbone):
        super(DefocusNet, self).__init__()

        # Physics-based defocus map generator
        self.defocus_generator = DefocusMapGenerator()

        # XceptionNet classifier
        self.classifier = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        
    def normalize_per_sample(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0]
        max_val = x_flat.max(dim=1, keepdim=True)[0]
        norm = (x_flat - min_val) / (max_val - min_val + 1e-8)
        return norm.view(B, C, H, W)
    
    def forward(self, x):
        physics_defocus_map_gt, _, _, timer = self.defocus_generator(x, x.device)

        print("timers:", timer)  # print per-stage performance
        # CSV append
        with open("timer_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for stage, vals in timer.items():
                writer.writerow([stage, vals["time_ms"], vals["max_mem_bytes"]])
        physics_defocus_map_gt = F.interpolate(physics_defocus_map_gt.detach(), size=(299, 299), mode="bilinear", align_corners=False)
        physics_defocus_map_gt = self.normalize_per_sample(physics_defocus_map_gt)
         # If single channel, expand to 3 channels for ImageNet models
        if physics_defocus_map_gt.shape[1] == 1:
            physics_defocus_map_gt = physics_defocus_map_gt.repeat(1, 3, 1, 1)

        print(physics_defocus_map_gt.max(), physics_defocus_map_gt.min())
        class_input = physics_defocus_map_gt
        # print(class_input.shape)
        class_output = self.classifier(class_input)

        return physics_defocus_map_gt, class_output


class MultiBranchDefocusNet(nn.Module):
    def __init__(self, num_classes, rgb_backbone, defocus_backbone):
        super(MultiBranchDefocusNet, self).__init__()

        self.rgb_backbone = timm.create_model(rgb_backbone, pretrained=True, num_classes=0, global_pool='avg')
        self.defocus_backbone = timm.create_model(defocus_backbone, pretrained=False, in_chans=1, num_classes=0, global_pool='avg')

        rgb_feat_dim = self.rgb_backbone.num_features
        defocus_feat_dim = self.defocus_backbone.num_features

        self.classifier = nn.Linear(rgb_feat_dim, num_classes)

    def forward(self, rgb, defocus_map):
        rgb_feat = self.rgb_backbone(rgb)              # (batch, rgb_feat_dim)
        defocus_feat = self.defocus_backbone(defocus_map)  # (batch, defocus_feat_dim)
        # print(rgb_feat.shape, defocus_feat.shape)
        fused_feat = rgb_feat * defocus_feat

        output = self.classifier(fused_feat)                     # (batch, num_classes)

        return output
