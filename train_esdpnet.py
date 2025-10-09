
import os
import cv2
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import KFold

from model_esdpnet import ESDPNet          # 保持原模型


# ───────────────────── CLI 参数 ─────────────────────
def get_argparse():
    parser = argparse.ArgumentParser("ES-DPNet training")
    parser.add_argument("--gpu", default="0", help="CUDA device id(s)")
    parser.add_argument("--root-path", type=str, default="/dataset",
                        help="ImageFolder 数据根目录")
    parser.add_argument("--exp-name", type=str, default="", help="实验标识（保存文件夹名后缀）")

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="开启混合精度")
    return parser


# ───────────────────── 数据集定义 ─────────────────────
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_hog = transforms.Compose([
    transforms.Resize((128, 128)),
])

def extract_hog_features(img: Image.Image):
    hog = cv2.HOGDescriptor(
        _winSize=(128 // 8 * 8, 128 // 8 * 8),
        _blockSize=(8 * 8, 8 * 8),
        _blockStride=(4 * 4, 4 * 4),
        _cellSize=(8, 8), _nbins=12
    )
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return hog.compute(gray)

class HOGImageDataset(Dataset):
    def __init__(self, root, transform_img=None, transform_hog=None, batch_size=500):
        self.dataset = datasets.ImageFolder(root, transform=transform_img)
        self.transform_hog = transform_hog
        self.batch_size = batch_size
        self._build_hog_bank()

    def _build_hog_bank(self):
        feats, labels = [], []
        for i in range(0, len(self.dataset.imgs), self.batch_size):
            for img_path, lab in self.dataset.imgs[i:i+self.batch_size]:
                img = Image.open(img_path).convert("RGB")
                img_hog = self.transform_hog(img) if self.transform_hog else img
                feats.append(extract_hog_features(img_hog))
                labels.append(lab)
        self.hog = np.array(feats, dtype=np.float32).reshape(len(feats), -1)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb_img, _ = self.dataset[idx]
        hog_feat = torch.from_numpy(self.hog[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb_img, hog_feat, label


# ───────────────────── 评估函数 ─────────────────────
@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, hogs, labels in loader:
        imgs, hogs, labels = imgs.to(device), hogs.to(device), labels.to(device)
        preds = model(imgs, hogs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ───────────────────── 训练主循环 ─────────────────────
def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = opt.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"ESDPNet_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    full_ds = HOGImageDataset(opt.root_path, transform_img=transform_image,
                              transform_hog=transform_hog)
    kfold = KFold(n_splits=5, shuffle=True)

    best_global_acc, best_global_path = 0.0, ""
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(full_ds), 1):
        print(f"\n===== Fold {fold}/5 =====")
        tr_loader = DataLoader(Subset(full_ds, tr_idx),
                               batch_size=opt.batch_size, shuffle=True,
                               num_workers=opt.num_workers)
        va_loader = DataLoader(Subset(full_ds, va_idx),
                               batch_size=opt.batch_size, shuffle=False,
                               num_workers=opt.num_workers)

        model = ESDPNet(num_classes=7,
                        hog_input_dim=full_ds[0][1].numel()).to(device)

        optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scaler = GradScaler(enabled=opt.amp)
        criterion = nn.CrossEntropyLoss()

        best_fold_acc, best_fold_path = 0.0, ""

        for epoch in range(1, opt.num_epoch + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(tr_loader, desc=f"Fold{fold} Epoch{epoch}/{opt.num_epoch}", leave=False)
            for imgs, hogs, labels in pbar:
                imgs, hogs, labels = imgs.to(device), hogs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast(enabled=opt.amp):
                    outputs = model(imgs, hogs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))

            val_acc = evaluate_accuracy(model, va_loader, device)
            print(f"Epoch {epoch:03d}/{opt.num_epoch}  Val Acc: {val_acc*100:.2f}%")

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_path = os.path.join(
                    ckpt_dir, f"fold{fold}_best_{val_acc:.4f}.pth")
                torch.save(model.state_dict(), best_fold_path)
                print(f"  → New best for fold, saved to {best_fold_path}")

        print(f"Fold {fold} best Val Acc = {best_fold_acc*100:.2f}%")

        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc
            best_global_path = best_fold_path

    print(f"\nOverall best model: {best_global_path}  "
          f"Acc = {best_global_acc*100:.2f}%")


# ───────────────────── main ─────────────────────
if __name__ == "__main__":
    args = get_argparse().parse_args()
    train(args)
