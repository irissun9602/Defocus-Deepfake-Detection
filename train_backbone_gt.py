import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.metrics import roc_auc_score
import numpy as np
import tqdm

from dataset.FF import FaceForensicsDataset  # 사용자 정의 FF++ 데이터셋 클래스
from dataset.CiFake import CiFakeDataset, get_train_val_datasets, get_test_dataset
from dataset.diffusiondb import DiffusiondbDataset
from dataset.COCO import COCODataset
import timm
from util.metrics import compute_metrics_bce, save_metrics_to_csv

from model.DefocusNet_backbone_defocus_gt import DefocusNet  # 위에서 정의한 모델을 불러온다고 가정
import random
import numpy as np
import argparse

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------- 커맨드라인 인자 설정 ---------------------------
parser = argparse.ArgumentParser(description="Train Defocus_GT with configurable backbone and fake type")
parser.add_argument("--backbone", type=str, default="efficientnet_b4", help="Backbone model (e.g. resnet50, xception, efficientnet_b4)")
parser.add_argument("--fake_type", type=str, default="Deepfakes", help="Fake type (e.g. Deepfakes, Face2Face, FaceSwap, NeuralTextures)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()


# --------------------------- 설정 ---------------------------
FF_RGB_ROOT = "/media/NAS/DATASET/faceforensics++/Dec2020/v1face_data/data_c0"
FAKE_TYPE = args.fake_type  # 또는 Face2Face, FaceSwap 등
EPOCHS = args.epochs
BATCH_SIZE = 64
BACKBONE = args.backbone  # 사용할 백본 모델 (예: xception, resnet50 등)
# WEIGHT_PATH = f"defocus_gt_{BACKBONE}_{FAKE_TYPE}.pth"
WEIGHT_PATH = "./weights/defocus_gt_xception_Deepfakes.pth"
SAVE_MAPS_DIR = "./defocus_gt_maps"
os.makedirs(SAVE_MAPS_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# --------------------------- 데이터셋 1 ---------------------------
train_dataset = FaceForensicsDataset(root_dir=FF_RGB_ROOT, phase="train", fake_types=[FAKE_TYPE])
val_dataset = FaceForensicsDataset(root_dir=FF_RGB_ROOT, phase="val", fake_types=[FAKE_TYPE])
test_dataset = FaceForensicsDataset(root_dir=FF_RGB_ROOT, phase="test", fake_types=[FAKE_TYPE])
# tt_real = Subset(train_dataset, np.arange(0, 1000))
# tt_fake = Subset(train_dataset, np.arange(700000, 701000))
# tt = torch.utils.data.ConcatDataset([tt_real, tt_fake])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --------------------------- 데이터셋 2 ---------------------------
# test_dir = "./dataset/instagram"
# test_dataset = DiffusiondbDataset(test_dir)
# # 경로 설정
# train_dir = "/media/NAS/DATASET/cifake/train"
# test_dir = "/media/NAS/DATASET/cifake/test"

# # train/val split
# train_dataset, val_dataset = get_train_val_datasets(train_dir, transform=transform)

# # test dataset
# test_dataset = get_test_dataset(test_dir, transform=transform)

# from collections import Counter

# print("Train:", Counter([label for _, label in train_dataset.samples]))
# print("Val:",   Counter([label for _, label in val_dataset.samples]))
# print("Test:",  Counter([label for _, label in test_dataset.samples]))

# # DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4 )
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


# --------------------------- 모델 정의 ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DefocusNet(num_classes=1, backbone=BACKBONE)
model = model.to(device)

print(f"✅ Using backbone: {BACKBONE}")
print(f"✅ Model class: {model.classifier.__class__.__name__}")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------- 학습 루프 ---------------------------
start_epoch = 0
best_val_acc = 0.0

# if os.path.exists(WEIGHT_PATH):
#     print(f"🔄 Resuming from {WEIGHT_PATH}")
#     checkpoint = torch.load(WEIGHT_PATH)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
for epoch in range(EPOCHS):
    model.train()
    total_correct, total_samples = 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for i, (images, labels, paths) in enumerate(pbar):
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        defocus_map, outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # ✅ tqdm에 실시간 loss와 acc 표시
        current_acc = total_correct / total_samples
        pbar.set_postfix({"Loss": loss.item(), "Acc": current_acc})

        # ✅ Defocus Map + 원본 이미지 저장 (일부 배치만 저장)
        if i % 10 == 0:
            for j in range(min(4, defocus_map.size(0))):
                defocus_img = defocus_map[j].cpu()
                rgb_img = (images[j].cpu() * 0.5 + 0.5).clamp(0, 1)  # Normalize 해제

                combined = make_grid([rgb_img, defocus_img.expand_as(rgb_img)], nrow=2)
                img_path = paths[j].replace("/", "_").replace(":", "_")  # 파일명 안전하게
                save_path = os.path.join(SAVE_MAPS_DIR, f"epoch{epoch+1}_batch{i}_sample{j}_{img_path}.png")
                save_image(combined, save_path)

    acc = total_correct / total_samples
    auc = roc_auc_score(all_labels, all_preds)
    print(f"[Train] Acc: {acc:.4f}, AUC: {auc:.4f}")
    # ✅ 메트릭 저장
    metrics = compute_metrics_bce(all_preds, all_labels)
    save_metrics_to_csv("train", BACKBONE, WEIGHT_PATH, FAKE_TYPE, loss, metrics, "defocus_gt_results.csv")

    # ---------------- 검증 ----------------
    model.eval()
    total_correct, total_samples = 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm.tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            defocus_map, outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_acc = total_correct / total_samples
            val_loss = criterion(outputs, labels)
            pbar.set_postfix({"Loss": val_loss, "Acc": current_acc})
            
    val_acc = total_correct / total_samples
    val_auc = roc_auc_score(all_labels, all_preds)
    print(f"[Val] Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
    # ✅ 메트릭 저장
    metrics = compute_metrics_bce(all_preds, all_labels)
    save_metrics_to_csv("val", BACKBONE, WEIGHT_PATH, FAKE_TYPE, val_loss, metrics, "defocus_gt_results.csv")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, WEIGHT_PATH)
        print("Best model saved.")

# --------------------------- 테스트 ---------------------------
def test(model, loader):
    checkpoint = torch.load(WEIGHT_PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])  # ✅ 변경된 부분
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.eval()
    all_preds, all_labels = [], []
    total_correct, total_samples = 0, 0
    pbar = tqdm.tqdm(loader, desc="Test")
    with torch.no_grad():
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            defocus_map, outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_acc = total_correct / total_samples
            test_loss = criterion(outputs, labels)
            pbar.set_postfix({"Loss": test_loss, "Acc": current_acc})

    test_auc = roc_auc_score(all_labels, all_preds)
    test_acc = total_correct / total_samples
    print(f"[Test] Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    # ✅ 메트릭 저장
    metrics = compute_metrics_bce(all_preds, all_labels)
    save_metrics_to_csv("test", BACKBONE, WEIGHT_PATH, FAKE_TYPE, test_loss, metrics, "defocus_gt_results.csv")
    return test_acc, test_auc

# --------------------------- 실행 ---------------------------
test(model, test_loader)
