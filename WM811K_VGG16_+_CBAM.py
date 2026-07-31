import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context
 
# ==========================================
# 1. DATA CLEANING, FILTERING & ENCODING
# ==========================================
print("Loading raw WM-811K dataset...")
df = pd.read_pickle("WM811K.pkl")
 
def clean_wafer_label(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0: return "none"
        item = x
        while isinstance(item, (list, np.ndarray)) and len(item) > 0:
            item = item[0]
        return str(item)
    return str(x) if pd.notna(x) else "none"
 
print("Processing and cleansing nested dictionary patterns...")
df['clean_label'] = df['failureType'].apply(clean_wafer_label)
df = df[df['clean_label'].notna() & df['waferMap'].notna()]
 
valid_classes = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                 'Loc', 'Random', 'Scratch', 'Near-full']
df_filtered = df[df['clean_label'].isin(valid_classes)].copy()
 
label_encoder = LabelEncoder()
df_filtered['encoded_label'] = label_encoder.fit_transform(df_filtered['clean_label'])
num_classes = len(label_encoder.classes_)
 
print(f"Filtering Complete! Total wafers retained: {len(df_filtered)}")
class_mapping = dict(zip(label_encoder.classes_, range(num_classes)))
print(f"Class Target Index Mapping: {class_mapping}")
 
 
# ==========================================
# 2. STRATIFIED SPLITS (90 / 5 / 5)
# ==========================================
indices = np.arange(len(df_filtered))
labels  = df_filtered['encoded_label'].values
 
train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, labels, test_size=0.10, stratify=labels, random_state=42
)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
 
train_df = df_filtered.iloc[train_idx].copy()
val_df   = df_filtered.iloc[val_idx].copy()
test_df  = df_filtered.iloc[test_idx].copy()
 
print(f"Dataset Scale: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
 
 
# ==========================================
# 3. PYTORCH DATASETS & TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])
 
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
 
class WaferVGG16Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.wafer_maps = dataframe['waferMap'].values
        self.labels     = dataframe['encoded_label'].values
        self.transform  = transform
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        wafer_grid = np.array(self.wafer_maps[idx])
        label      = int(self.labels[idx])
        resized    = resize(wafer_grid, (224, 224), order=0,
                            preserve_range=True,
                            anti_aliasing=False).astype(np.uint8)
        rgb_wafer  = np.stack([resized] * 3, axis=-1)
        if self.transform:
            rgb_wafer = self.transform(rgb_wafer)
        return rgb_wafer, torch.tensor(label, dtype=torch.long)
 
train_dataset = WaferVGG16Dataset(train_df, transform=train_transform)
val_dataset   = WaferVGG16Dataset(val_df,   transform=val_test_transform)
test_dataset  = WaferVGG16Dataset(test_df,  transform=val_test_transform)
 
 
# ==========================================
# 4. WEIGHTED RANDOM SAMPLER
# ==========================================
counts_dict   = train_df['encoded_label'].value_counts().to_dict()
class_weights = np.zeros(num_classes)
for label in range(num_classes):
    if label in counts_dict and counts_dict[label] > 0:
        class_weights[label] = 1.0 / counts_dict[label]
 
sample_weights = [class_weights[label] for label in train_df['encoded_label']]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)
 
BATCH_SIZE   = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
 
 
# ==========================================
# 5. CBAM — CONVOLUTIONAL BLOCK ATTENTION MODULE
# ==========================================
class ChannelAttention(nn.Module):
    """
    Channel attention: learns WHICH feature maps to emphasise.
    Uses both avg-pool and max-pool to capture global context,
    passes through a shared MLP, then applies sigmoid gate.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale
 
 
class SpatialAttention(nn.Module):
    """
    Spatial attention: learns WHERE in the feature map to focus.
    Concatenates channel-wise avg and max projections, then applies
    a 7x7 conv + sigmoid to produce a spatial gate.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat  = torch.cat([avg_out, max_out], dim=1)
        scale   = self.sigmoid(self.conv(concat))
        return x * scale
 
 
class CBAM(nn.Module):
    """
    Full CBAM block: Channel Attention -> Spatial Attention (sequential).
    Plugged in after VGG16 feature extractor before the classifier head.
    in_channels = 512 (VGG16 last conv output channels).
    reduction_ratio = 16 (standard CBAM default).
    """
    def __init__(self, in_channels=512, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)
 
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
 
 
# ==========================================
# 6. VGG16 + CBAM MODEL DEFINITION
# ==========================================
class VGG16WithCBAM(nn.Module):
    """
    VGG16 backbone with CBAM attention inserted between the feature
    extractor and the classification head.
 
    Architecture:
        VGG16 features (conv layers)
            ↓
        CBAM (channel + spatial attention)
            ↓
        AdaptiveAvgPool2d → 7x7
            ↓
        Flatten
            ↓
        Dense(4096) → ReLU → Dropout(0.5)
        Dense(4096) → ReLU → Dropout(0.5)
        Dense(num_classes)
    """
    def __init__(self, num_classes, reduction_ratio=16):
        super().__init__()
        base        = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = base.features          # VGG16 conv blocks
        self.cbam     = CBAM(in_channels=512, reduction_ratio=reduction_ratio)
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        # Unfreeze all backbone weights for full fine-tuning
        for param in self.features.parameters():
            param.requires_grad = True
 
    def forward(self, x):
        x = self.features(x)     # VGG16 feature extraction
        x = self.cbam(x)         # CBAM attention
        x = self.avgpool(x)      # spatial pooling
        x = self.classifier(x)   # classification head
        return x
 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Target Hardware: {device}")
 
model     = VGG16WithCBAM(num_classes=num_classes, reduction_ratio=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=2
)
 
total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
 
 
# ==========================================
# 7. TRAINING LOOP
# ==========================================
EPOCHS                 = 15
best_val_loss          = float('inf')
patience_counter       = 0
early_stopping_patience = 3
 
train_losses = []
val_losses   = []
 
print("\nStarting VGG16 + CBAM fine-tuning loop...")
 
for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Step Loss: {loss.item():.4f}")
 
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
 
    # --- Validate ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss    = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)
 
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
 
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
 
    scheduler.step(epoch_val_loss)
 
    if epoch_val_loss < best_val_loss:
        best_val_loss    = epoch_val_loss
        torch.save(model.state_dict(), 'best_vgg16_cbam_wafer_model.pth')
        patience_counter = 0
        print("  Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
 
 
# ==========================================
# CHART 1: TRAINING LOSS HISTORY
# ==========================================
print("\nGenerating Training Loss History Chart...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses,
         marker='o', color='#0ea5e9', label='Training Loss',   linewidth=2)
plt.plot(range(1, len(val_losses)+1),   val_losses,
         marker='s', color='#f43f5e', label='Validation Loss', linewidth=2)
plt.title('VGG16 + CBAM Cross-Entropy Loss Optimization Trace',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, len(train_losses)+1))
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('training_metrics_history.png', dpi=300)
plt.close()
print("Saved: training_metrics_history.png")
 
 
# ==========================================
# 8. TEST EVALUATION
# ==========================================
print("\nLoading best model weights for test evaluation...")
model.load_state_dict(torch.load('best_vgg16_cbam_wafer_model.pth'))
model.eval()
 
all_preds   = []
all_targets = []
 
with torch.no_grad():
    for images, targets in test_loader:
        images  = images.to(device)
        outputs = model(images)
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
 
all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)
 
unique_test_classes  = np.unique(all_targets)
present_class_names  = label_encoder.inverse_transform(unique_test_classes)
 
report_dict = classification_report(
    all_targets, all_preds,
    labels=unique_test_classes,
    target_names=present_class_names,
    output_dict=True
)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    all_targets, all_preds,
    labels=unique_test_classes,
    target_names=present_class_names
))
 
cm = confusion_matrix(all_targets, all_preds, labels=unique_test_classes)
 
 
# ==========================================
# CHART 2: CONFUSION MATRIX HEATMAP
# ==========================================
print("\nGenerating Confusion Matrix Heatmap...")
plt.figure(figsize=(11, 9))
sns.set_theme(style='white')
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=present_class_names,
    yticklabels=present_class_names,
    cbar_kws={'label': 'Wafer Inspection Count'},
    annot_kws={'size': 11, 'weight': 'bold'}
)
plt.title('VGG16 + CBAM Silicon Defect Confusion Matrix',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Defect Classification', fontsize=12, labelpad=10)
plt.ylabel('True Defect Label',               fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0,              fontsize=10)
plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=300)
plt.close()
print("Saved: final_confusion_matrix.png")
 
 
# ==========================================
# CHART 3: GROUPED BAR CHART
#          Precision / Recall / F1 / Accuracy (%)
# ==========================================
print("\nGenerating Performance Metrics Bar Chart...")
 
classes_list    = list(present_class_names)
precisions      = [report_dict[c]['precision'] * 100 for c in classes_list]
recalls         = [report_dict[c]['recall']    * 100 for c in classes_list]
f1_scores       = [report_dict[c]['f1-score']  * 100 for c in classes_list]
global_accuracy = report_dict['accuracy'] * 100
 
# Per-class accuracy = diagonal / row sum
per_class_acc = []
for i, cls_idx in enumerate(unique_test_classes):
    row_sum = cm[i].sum()
    per_class_acc.append((cm[i, i] / row_sum * 100) if row_sum > 0 else 0.0)
 
x_idx     = np.arange(len(classes_list))
bar_width  = 0.20
 
fig, ax = plt.subplots(figsize=(16, 7))
sns.set_theme(style='whitegrid')
 
ax.bar(x_idx - 1.5*bar_width, precisions,    width=bar_width,
       label='Precision', color='#0ea5e9', alpha=0.87)
ax.bar(x_idx - 0.5*bar_width, recalls,       width=bar_width,
       label='Recall',    color='#10b981', alpha=0.87)
ax.bar(x_idx + 0.5*bar_width, f1_scores,     width=bar_width,
       label='F1-Score',  color='#f43f5e', alpha=0.87)
ax.bar(x_idx + 1.5*bar_width, per_class_acc, width=bar_width,
       label='Accuracy',  color='#8B5CF6', alpha=0.87)
 
ax.axhline(y=global_accuracy, color='#64748b', linestyle='--', linewidth=1.5,
           label=f'Global Accuracy ({global_accuracy:.2f}%)')
ax.axhline(y=80, color='gray', linestyle=':', linewidth=1.0,
           alpha=0.6, label='80% threshold')
 
ax.set_title('VGG16 + CBAM Classification Metrics per Defect Class (%)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Silicon Wafer Defect Categories', fontsize=12, labelpad=12)
ax.set_ylabel('Performance Score (%)',           fontsize=12, labelpad=12)
ax.set_xticks(x_idx)
ax.set_xticklabels(classes_list, rotation=35, ha='right', fontsize=11)
ax.set_ylim(0, 110)
ax.legend(loc='lower left', fontsize=10, frameon=True,
          facecolor='#ffffff', edgecolor='#cbd5e1')
plt.tight_layout()
plt.savefig('final_metrics_comparison.png', dpi=300)
plt.close()
print("Saved: final_metrics_comparison.png")
 
 
# ==========================================
# CHART 4: PER-CLASS METRICS TABLE (%)
#          Precision / Recall / F1 / Accuracy
#          Colour-coded: green>=80, amber>=60, red<60
# ==========================================
print("\nGenerating Per-class Metrics Table...")
 
col_labels = ['Class', 'Precision %', 'Recall %', 'F1 %', 'Accuracy %']
cell_data  = []
for i, cls in enumerate(classes_list):
    cell_data.append([
        cls,
        f"{precisions[i]:.1f}%",
        f"{recalls[i]:.1f}%",
        f"{f1_scores[i]:.1f}%",
        f"{per_class_acc[i]:.1f}%",
    ])
 
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis('off')
 
table = ax.table(
    cellText  = cell_data,
    colLabels = col_labels,
    cellLoc   = 'center',
    loc       = 'center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.9)
 
# Header row style
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#0ea5e9')
    table[0, j].set_text_props(color='white', fontweight='bold')
 
# Colour-code metric cells
score_cols = {
    1: precisions,
    2: recalls,
    3: f1_scores,
    4: per_class_acc,
}
for i in range(len(classes_list)):
    # Class name column — neutral
    table[i+1, 0].set_facecolor('#F5F5F5')
    for col_idx, scores in score_cols.items():
        val = scores[i]
        if   val >= 80: bg = '#E6F4EA'   # green
        elif val >= 60: bg = '#FFF8E1'   # amber
        else:           bg = '#FDECEA'   # red
        table[i+1, col_idx].set_facecolor(bg)
 
plt.title('Per-class metrics table — VGG16 + CBAM (%)',
          fontsize=13, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('per_class_metrics_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: per_class_metrics_table.png")
 
# Also print to console
print("\nPer-class metrics summary (%):")
print(f"{'Class':<14} {'Precision':>12} {'Recall':>10} {'F1':>8} {'Accuracy':>12}")
print("-" * 60)
for i, cls in enumerate(classes_list):
    print(f"{cls:<14} {precisions[i]:>11.1f}% {recalls[i]:>9.1f}% "
          f"{f1_scores[i]:>7.1f}% {per_class_acc[i]:>11.1f}%")
print(f"\nGlobal accuracy: {global_accuracy:.2f}%")
 
 
# ==========================================
# CHART 5: PIPELINE HYPERPARAMETER SUMMARY
# ==========================================
print("\nGenerating Pipeline Summary Card...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
 
pipeline_text = (
    "===========================================================\n"
    "     SILICON WAFER DEFECT DETECTION — VGG16 + CBAM PIPELINE\n"
    "===========================================================\n\n"
    f"  Hardware Accelerator   : NVIDIA L40 (48GB VRAM)\n"
    f"  Dataset Scale          : {len(df_filtered):,} expert-verified records\n"
    f"  Split                  : 90% Train | 5% Val | 5% Test (Stratified)\n"
    f"  Input Resolution       : 224 x 224 x 3 RGB\n"
    f"  Pixel Normalisation    : [0.0, 1.0]\n"
    f"  Imbalance Mitigation   : Weighted Random Sampler\n"
    f"  Batch Size             : {BATCH_SIZE}\n"
    f"  Backbone               : VGG16 (full fine-tuning)\n"
    f"  Attention Module       : CBAM (Channel + Spatial, ratio=16)\n"
    f"  Classifier Head        : Linear(4096) x2 → Dropout(0.5) → {num_classes} classes\n"
    f"  Max Epochs             : {EPOCHS}\n"
    f"  Optimiser              : Adam (lr=1e-5)\n"
    f"  Loss Function          : Cross-Entropy\n"
    f"  LR Scheduler           : ReduceLROnPlateau (patience=2, factor=0.1)\n"
    f"  Early Stopping         : patience={early_stopping_patience} epochs\n\n"
    "==========================================================="
)
ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='#f8fafc', edgecolor='#cbd5e1'))
plt.tight_layout()
plt.savefig('preprocessing_and_parameters.png', dpi=300)
plt.close()
print("Saved: preprocessing_and_parameters.png")
 
print("\n" + "=" * 55)
print("ALL SAVED FILES")
print("=" * 55)
for fname in [
    'training_metrics_history.png',
    'final_confusion_matrix.png',
    'final_metrics_comparison.png',
    'per_class_metrics_table.png',
    'preprocessing_and_parameters.png',
    'best_vgg16_cbam_wafer_model.pth',
]:
    status = '✓' if os.path.exists(fname) else '✗ not yet saved'
    print(f"  {status}  {fname}")
print("=" * 55)
print("Pipeline complete.")