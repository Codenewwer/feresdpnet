import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import wandb

from model_esdpnet import ESDPNet
from utils.metrics import classifier_evaluate

os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
try:
    wandb.login()
except wandb.errors.UsageError:
    print("W&B 已登录或 API 密钥已配置。")

wandb.init(project="", entity='')
print(f"W&B 本地日志保存在：{wandb.run.dir}")

# 开启 cuDNN 自动调优
torch.backends.cudnn.benchmark = True

# 数据预处理
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
transform_hog = transforms.Compose([
    transforms.Resize((128, 128)),
])


def extract_hog_features(image):
    hog = cv2.HOGDescriptor(
        _winSize=(128 // 8 * 8, 128 // 8 * 8),  # (128,128)
        _blockSize=(8 * 8, 8 * 8),  # (64,64)
        _blockStride=(4 * 4, 4 * 4),  # (16,16)
        _cellSize=(8, 8),
        _nbins=12
    )
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return hog.compute(gray).flatten()


class HOGImageDataset(Dataset):
    def __init__(self, root, transform_image=None, transform_hog=None, batch_size=500):
        self.dataset = datasets.ImageFolder(root, transform=transform_image)
        self.transform_hog = transform_hog
        self.hog_features, self.labels = [], []
        imgs = self.dataset.imgs
        print(f"Loading HOG features for {root} with {len(imgs)} images...")
        for i in range(0, len(imgs), batch_size):
            feats, labs = [], []
            current_batch_imgs = imgs[i:i + batch_size]
            for path, lbl in current_batch_imgs:
                img = Image.open(path).convert('RGB')
                if self.transform_hog: img = self.transform_hog(img)
                try:
                    hog_feat = extract_hog_features(img)
                    feats.append(hog_feat)
                    labs.append(lbl)
                except cv2.error as e:
                    print(f"Error extracting HOG from {path}: {e}")
                    continue

            if feats:
                self.hog_features.append(np.vstack(feats).astype(np.float32))
                self.labels.append(np.array(labs))
            print(f"Processed {min(i + batch_size, len(imgs))} / {len(imgs)} images for HOG features in {root}.")

        if self.hog_features:
            self.hog_features = np.concatenate(self.hog_features, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            print(f"No images found or HOG extraction failed for {root}.")
            self.hog_features = np.array([], dtype=np.float32).reshape(0, 0)
            self.labels = np.array([], dtype=np.long)

        if self.hog_features.shape[0] > 0:
            print(f"HOG features shape for {root}: {self.hog_features.shape}")
        else:
            print(f"Warning: HOG features array is empty for {root}. Check your dataset path and image loading.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        hog = torch.tensor(self.hog_features[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, hog, lbl


# 标签映射
label_mapping = {
    0: "happy",
    1: "neutral",
    2: "sad",
    3: "angry",
    4: "disgust",
    5: "fear",
    6: "surprise"
}

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = HOGImageDataset('/public/home/train', transform_image, transform_hog,
                                batch_size=500)
test_dataset = HOGImageDataset('/public/home/test', transform_image, transform_hog,
                               batch_size=500)

train_loader = DataLoader(
    train_dataset,
    batch_size=32, shuffle=True, num_workers=1, pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

hog_dim = None
if train_dataset.hog_features.shape[0] > 0:
    hog_dim = train_dataset.hog_features.shape[1]
else:
    print("Warning: Training set HOG features are empty. HOG branch might not be initialized correctly.")

model = ESDPNet(num_classes=7, hog_input_dim=hog_dim).to(device)

# 尝试加载预训练模型
pretrained_model_path = '/public/home/pretrainbest_model.pth'
if os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from: {pretrained_model_path}")
    try:
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    except Exception as e:
        print(f"Warning: Failed to fully load pretrained weights: {e}")
else:
    print(f"Pretrained model not found at: {pretrained_model_path}. Starting training from scratch.")

for name, param in model.named_parameters():
    if any(seg in name for seg in ['conv1', 'bn1', 'layer0', 'layer1']):
        param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_accuracy = 0.0
local_best_model_path = 'best_model.pth'

torch.save(model.state_dict(), local_best_model_path)
print(f"Initial model state saved to: {local_best_model_path}")

for epoch in range(50):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (imgs, hogs, labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        hogs = hogs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs, hogs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Epoch {epoch + 1}/{30}, Batch {batch_idx + 1}/{len(train_loader)} - '
                  f'Train Loss: {loss.item():.4f}, Acc: {correct / total:.2%}')

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, hogs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            hogs = hogs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs, hogs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = val_loss / len(test_loader)
    val_acc = correct / total

    print(f'Epoch {epoch + 1}/30  Train L {train_loss:.4f} Acc {train_acc:.2%}  '
          f'Val L {val_loss:.4f} Acc {val_acc:.2%}')
    wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_acc,
               'val_loss': val_loss, 'val_accuracy': val_acc})

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), local_best_model_path)
        print(f'保存了新的最佳本地模型: {local_best_model_path}, 准确率: {best_val_accuracy:.2%}')

        artifact = wandb.Artifact(
            "emotion-recognition-model",
            type="model",
            description=f"Best model from epoch {epoch + 1} with validation accuracy {best_val_accuracy:.4f}"
        )
        artifact.add_file(local_best_model_path)
        wandb.log_artifact(artifact, aliases=["best", f"epoch_{epoch + 1}", f"acc_{best_val_accuracy:.4f}"])
        print(f"Uploaded best model artifact to WandB: {artifact.name}")

best_model_filename = local_best_model_path
print(f'训练结束。最佳验证准确率: {best_val_accuracy:.2%}')
print(f'最终本地保存的最佳模型文件路径: {best_model_filename}')


def plot_confusion_matrix_and_log(true_labels, predicted_labels, title, filename, label_map, normalize=False):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized[np.isnan(cm_normalized)] = 0
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=list(label_map.values()),
                    yticklabels=list(label_map.values()))
        plt.title(f"{title} (Percentages)")
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(label_map.values()),
                    yticklabels=list(label_map.values()))
        plt.title(f"{title} (Counts)")

    plt.xlabel("Predicted Label");
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    wandb.log({filename.replace('.png', ''): wandb.Image(filename)})
    print(f"Saved and logged: {filename}")


if os.path.exists(best_model_filename):
    model.load_state_dict(torch.load(best_model_filename))
    print(f"Loaded best model for final evaluation from: {best_model_filename}")
else:
    print(f"Error: Best model file not found at {best_model_filename}. Cannot perform final evaluation.")
    wandb.finish()
    exit()

model.eval()

print("\n--- Evaluating on Test Set ---")
all_preds_test, all_labels_test = [], []
with torch.no_grad():
    for imgs, hogs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        hogs = hogs.to(device, non_blocking=True)
        outputs = model(imgs, hogs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds_test.extend(preds)
        all_labels_test.extend(labels.numpy())

acc_test, wp_test, uar_test, waf_test, cm_test = classifier_evaluate(all_labels_test, all_preds_test)

print(f"Test Set Accuracy (ACC): {acc_test:.4f}")
print(f"Test Set Weighted Precision (WP): {wp_test:.4f}")
print(f"Test Set Unweighted Average Recall (UAR): {uar_test:.4f}")
print(f"Test Set Weighted Average F1 (WAF): {waf_test:.4f}")

wandb.log({
    "final_test_ACC": acc_test,
    "final_test_WP": wp_test,
    "final_test_UAR": uar_test,
    "final_test_WAF": waf_test
})

plot_confusion_matrix_and_log(all_labels_test, all_preds_test, "Test Set Confusion Matrix", "test_cm_counts.png",
                              label_mapping, normalize=False)
plot_confusion_matrix_and_log(all_labels_test, all_preds_test, "Test Set Confusion Matrix", "test_cm_percentages.png",
                              label_mapping, normalize=True)

print(
    f'\n所有评估完成。最终使用的最佳模型文件: {best_model_filename}，该模型在验证集上的最佳准确率: {best_val_accuracy:.2%}')

wandb.finish()