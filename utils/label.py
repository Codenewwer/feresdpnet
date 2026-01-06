import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import ast
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# -------------------- Model Definitions --------------------

# 1) ECA module
class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.eca = ECALayer(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        out = self.eca(out)
        return out

class esdpnet(nn.Module):
    def __init__(self, num_classes=7, hog_input_dim=None):
        super(esdpnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer0 = self._make_layer(16, 32, 1, True)
        self.layer1 = self._make_layer(32, 64, 1, True)
        self.layer2 = self._make_layer(64, 128, 1, True)
        self.layer3 = self._make_layer(128, 256, 1, True)
        self.layer4 = self._make_layer(256, 512, 1, True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cnn = nn.Linear(512, num_classes)
        # If HOG features exist
        if hog_input_dim:
            self.fc1_hog = nn.Linear(hog_input_dim, 512)
            self.fc2_hog = nn.Linear(512, 256)
            self.fc3_hog = nn.Linear(256, 128)
            self.fc4_hog = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample):
        layers = [ResidualBlock(in_channels, out_channels, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, hog_features=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out_cnn = self.fc_cnn(out)

        if hog_features is not None:
            x_hog = torch.relu(self.fc1_hog(hog_features))
            x_hog = torch.relu(self.fc2_hog(x_hog))
            x_hog = torch.relu(self.fc3_hog(x_hog))
            out_hog = self.fc4_hog(x_hog)
            return out_cnn + out_hog

        return out_cnn

# 3) VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)

# 4) ResNet34
class ResNet34(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 5) EfficientNetB0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# 6) ConvNeXtV2 (tiny) -- you mentioned "convnextv2_tiny" is in conv2.py
#    so let's assume we can just import it.
#    If you need to define it here, place the definition inline.
from conv2 import convnextv2_tiny
class ConvNextV2(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNextV2, self).__init__()
        self.convnext = convnextv2_tiny()

    def forward(self, x):
        return self.convnext(x)

# -------------------- Utilities --------------------
label_mapping = {
    0: "happy",
    1: "neutral",
    2: "sad",
    3: "angry",
    4: "disgust",
    5: "fear",
    6: "surprise"
}

def extract_hog_features(image):
    """Compute HOG features for a PIL image."""
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(64, 64),
        _blockStride=(16, 16),
        _cellSize=(8, 8),
        _nbins=9
    )
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    hog_feature = hog.compute(gray)
    return hog_feature.flatten()

class InferenceDataset(Dataset):
    """
    A simple Dataset that collects all valid images (*.png, *.jpg, etc.) in a directory.
    """
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return image, img_path

def collate_fn(batch):
    images = [item[0] for item in batch]
    img_paths = [item[1] for item in batch]
    return images, img_paths

def parse_subsubdir_name(folder_name):
    """
    Convert something like:
       "20240619_151433_Video_1_1" -> "votevideo1-1"
       "20240619_151641_Speech_1_1" -> "voteSpeech_1_1"
    If neither 'Video' nor 'Speech' is found, just prepend 'vote'.
    Adjust if your desired naming is slightly different.
    """
    parts = folder_name.split("_")
    if "Video" in parts:
        idx = parts.index("Video")
        # If next 2 indices exist
        if idx + 2 < len(parts):
            return f"votevideo{parts[idx+1]}-{parts[idx+2]}"
        else:
            # fallback if there's not enough parts
            return "votevideo" + "_".join(parts[idx+1:])
    elif "Speech" in parts:
        idx = parts.index("Speech")
        if idx + 2 < len(parts):
            return f"voteSpeech_{parts[idx+1]}_{parts[idx+2]}"
        else:
            return "voteSpeech_" + "_".join(parts[idx+1:])
    else:
        return "vote" + folder_name

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------- Model Configs --------------------
    # NOTE: Adjust the 'weight_path' entries below to point to wherever your .pth files actually reside.
    #       The code assumes these paths are valid.
    model_configs = {
        'esdpnet': {
            'model_class': esdpnet,
            'params': {'num_classes': 7, 'hog_input_dim': 14400},
            'weight_path': '/public/home/best_model_acc.pth',
            'image_transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]),
            'hog_transform': transforms.Resize((128, 128))
        },
        'vgg16': {
            'model_class': VGG16,
            'params': {'num_classes': 7},
            'weight_path': '/public/home/vgg16.pth',
            'image_transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        },
        'resnet34': {
            'model_class': ResNet34,
            'params': {'num_classes': 7},
            'weight_path': '/public/home/resnet34.pth',
            'image_transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        },
        'efficientnet': {
            'model_class': EfficientNetB0,
            'params': {'num_classes': 7},
            'weight_path': '/public/home/efficientnet.pth',
            'image_transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        },
        'convnextv2': {
            'model_class': ConvNextV2,
            'params': {'num_classes': 7},
            'weight_path': '/public/home/covnextv2.pth',
            'image_transform': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    }

    # -------------------- Load Models --------------------
    loaded_models = {}
    for model_name, config in model_configs.items():
        try:
            model = config['model_class'](**config['params']).to(device)

            if not os.path.exists(config['weight_path']):
                print(f"[WARNING] Weight file not found for {model_name}: {config['weight_path']}")
                continue

            checkpoint = torch.load(config['weight_path'], map_location=device)
            state_dict = checkpoint

            # ---- Adjust potential key mismatches for each model ----
            # VGG16:
            if model_name == 'vgg16':
                new_sd = {}
                for k, v in state_dict.items():
                    if k.startswith('features.') or k.startswith('classifier.'):
                        new_sd['vgg.' + k] = v
                    else:
                        new_sd[k] = v
                state_dict = new_sd

            # ResNet34:
            if model_name == 'resnet34':
                new_sd = {}
                for key, value in state_dict.items():
                    if (key.startswith("conv1.") or
                        key.startswith("bn1.") or
                        key.startswith("layer")):
                        new_key = "resnet." + key
                    else:
                        new_key = key
                    new_key = new_key.replace("fc.", "resnet.fc.")
                    new_sd[new_key] = value
                state_dict = new_sd

            # EfficientNetB0:
            if model_name == 'efficientnet':
                new_sd = {}
                for key, value in state_dict.items():
                    if key.startswith("features.") or key.startswith("classifier."):
                        new_sd["efficientnet." + key] = value
                    else:
                        new_sd[key] = value
                state_dict = new_sd

            # ConvNeXtV2:
            if model_name == 'convnextv2':
                new_sd = {}
                for k, v in state_dict.items():
                    if k.startswith('downsample_layers.') or k.startswith('stages.'):
                        new_k = 'convnext.' + k
                    elif k.startswith('norm.'):
                        new_k = k.replace('norm.', 'convnext.norm.')
                    elif k.startswith('head.'):
                        new_k = k.replace('head.', 'convnext.head.')
                    else:
                        new_k = 'convnext.' + k
                    new_sd[new_k] = v
                state_dict = new_sd

            # Load into model
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            loaded_models[model_name] = {
                'model': model,
                'image_transform': config['image_transform'],
                'hog_transform': config.get('hog_transform', None)
            }
            print(f"[INFO] {model_name} loaded successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {str(e)}")

    if not loaded_models:
        print("No models were loaded. Please check your weight paths.")
        return

    # Where to save the final classification results
    base_output_dir = "/public/home/votedata"
    os.makedirs(base_output_dir, exist_ok=True)

    # This is the directory containing all s72, s79, etc.
    root_dir = "/public/home/data"

    # -------------------- Process each subject directory --------------------
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue  # skip if not a directory


        subject_output_dir = os.path.join(base_output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)


        for subsubdir in os.listdir(subject_path):
            subsubdir_path = os.path.join(subject_path, subsubdir)
            if not os.path.isdir(subsubdir_path):
                continue


            vote_subdir_name = parse_subsubdir_name(subsubdir)

            # Create a dataset for all images in subsubdir_path
            dataset = InferenceDataset(subsubdir_path)
            if len(dataset) == 0:
                # no images found, skip
                continue

            loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

            model_output_dirs = {}
            for model_name in loaded_models:
                model_output_dirs[model_name] = os.path.join(subject_output_dir, model_name)
                os.makedirs(model_output_dirs[model_name], exist_ok=True)

            voting_output_dir = os.path.join(subject_output_dir, vote_subdir_name)
            os.makedirs(voting_output_dir, exist_ok=True)

            # We'll store predictions from all models for each image in a dictionary
            # so we can apply the same voting logic as before
            all_preds_history = []  # We'll accumulate a dictionary per batch, like in your code

            # -------------- Inference Loop --------------
            with torch.no_grad():
                for pil_images, img_paths in loader:
                    all_preds = {path: [] for path in img_paths}

                    # For each model, run inference
                    for model_name, model_info in loaded_models.items():
                        # Transform images
                        transformed_images = [model_info['image_transform'](img) for img in pil_images]
                        img_tensor = torch.stack(transformed_images).to(device)

                        # HOG features for esdpnet if needed
                        hog_features = None
                        if model_name == 'esdpnet' and model_info['hog_transform'] is not None:
                            hog_processed = [model_info['hog_transform'](img) for img in pil_images]
                            hog_features = [torch.tensor(extract_hog_features(himg), dtype=torch.float32)
                                            for himg in hog_processed]
                            hog_features = torch.stack(hog_features).to(device)

                        # Forward pass
                        if model_name == 'esdpnet':
                            outputs = model_info['model'](img_tensor, hog_features)
                        else:
                            outputs = model_info['model'](img_tensor)

                        preds = torch.argmax(outputs, dim=1).cpu().numpy()

                        # Record predictions
                        for path, pred_label in zip(img_paths, preds):
                            all_preds[path].append(pred_label)

                    all_preds_history.append(all_preds)

            # -------------- After inference, distribute images --------------
            # We replicate your logic:
            # 1) Save each image into the subfolder for each model and its predicted emotion
            # 2) Perform the voting mechanism and save under voting folder if conditions are met
            #    Conditions:
            #      - if a label has >= 3 votes
            #         OR
            #      - if the first model's prediction appears >= 2 times among the 5 models
            #    then copy to final vote folder

            # Our label_mapping is the same
            # For clarity, track the model order
            model_order = ['esdpnet', 'vgg16', 'resnet34', 'efficientnet', 'convnextv2']

            for batch_preds_dict in all_preds_history:
                for img_path, preds_list in batch_preds_dict.items():
                    img_name = os.path.basename(img_path)

                    # -- Step 1: Save each model's classification result
                    for idx, pred_idx in enumerate(preds_list):
                        model_name = model_order[idx]
                        if model_name != 'esdpnet':
                            continue
                        emotion = label_mapping.get(pred_idx, "unknown")
                        # Create subdir subject_output_dir/model_name/emotion
                        model_emotion_dir = os.path.join(subject_output_dir, model_name, emotion)
                        os.makedirs(model_emotion_dir, exist_ok=True)

                        dest_model_img_path = os.path.join(model_emotion_dir, img_name)
                        # Copy only if not already copied
                        if not os.path.exists(dest_model_img_path):
                            shutil.copy(img_path, dest_model_img_path)

                    # -- Step 2: Voting
                    cnt = Counter(preds_list)
                    # Condition 1: if any label is voted >=3
                    # Condition 2: if count of first-model's label >=2
                    first_pred = preds_list[0]
                    if max(cnt.values()) >= 3:
                        final_pred = cnt.most_common(1)[0][0]
                        condition_met = True
                    elif cnt[first_pred] >= 2:
                        final_pred = first_pred
                        condition_met = True
                    else:
                        condition_met = False

                    if condition_met:
                        final_emotion = label_mapping.get(final_pred, "unknown")
                        vote_emotion_dir = os.path.join(voting_output_dir, final_emotion)
                        os.makedirs(vote_emotion_dir, exist_ok=True)

                        dest_voting_img_path = os.path.join(vote_emotion_dir, img_name)
                        if not os.path.exists(dest_voting_img_path):
                            shutil.copy(img_path, dest_voting_img_path)

    print("All classification and voting completed.")

if __name__ == "__main__":
    main()