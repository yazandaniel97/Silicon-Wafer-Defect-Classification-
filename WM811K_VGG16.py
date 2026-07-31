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
import ssl

# --- VISUALIZATION LIBRARIES ---
import matplotlib.pyplot as plt
import seaborn as sns

# Fix local certificate validation issues for weight downloads inside cluster
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. DATA CLEANING, COMPLETE FILTERING & ENCODING
# ==========================================
print("📦 Loading raw WM-811K dataset...")
df = pd.read_pickle("WM811K.pkl") 

def clean_wafer_label(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0: return "none"
        item = x
        while isinstance(item, (list, np.ndarray)) and len(item) > 0:
            item = item[0]
        return str(item)
    return str(x) if pd.notna(x) else "none"

print("🧹 Processing and cleansing nested dictionary patterns...")
df['clean_label'] = df['failureType'].apply(clean_wafer_label)
df = df[df['clean_label'].notna() & df['waferMap'].notna()]

valid_classes = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
df_filtered = df[df['clean_label'].isin(valid_classes)].copy()

label_encoder = LabelEncoder()
df_filtered['encoded_label'] = label_encoder.fit_transform(df_filtered['clean_label'])
num_classes = len(label_encoder.classes_)

print(f"✅ Filtering Complete! Total wafers retained (Defects + None): {len(df_filtered)}")
class_mapping = dict(zip(label_encoder.classes_, range(num_classes)))
print(f"📌 Class Target Index Mapping: {class_mapping}")

# ==========================================
# 2. FULL-SCALE STRATIFIED SPLITS (NO DOWN-SAMPLING)
# ==========================================
indices = np.arange(len(df_filtered))
labels = df_filtered['encoded_label'].values

train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, labels, test_size=0.10, stratify=labels, random_state=42
)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

train_df = df_filtered.iloc[train_idx].copy()
val_df = df_filtered.iloc[val_idx].copy()
test_df = df_filtered.iloc[test_idx].copy()

print(f"📊 Global Dataset Scale: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

# ==========================================
# 3. HIGH-EFFICIENCY PYTORCH DATASETS
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
        self.labels = dataframe['encoded_label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        wafer_grid = np.array(self.wafer_maps[idx])
        label = int(self.labels[idx])
        
        resized = resize(wafer_grid, (224, 224), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        rgb_wafer = np.stack([resized] * 3, axis=-1)
        
        if self.transform:
            rgb_wafer = self.transform(rgb_wafer)
        return rgb_wafer, torch.tensor(label, dtype=torch.long)

train_dataset = WaferVGG16Dataset(train_df, transform=train_transform)
val_dataset = WaferVGG16Dataset(val_df, transform=val_test_transform)
test_dataset = WaferVGG16Dataset(test_df, transform=val_test_transform)

# ==========================================
# 4. ROBUST WEIGHTED RANDOM SAMPLER
# ==========================================
counts_dict = train_df['encoded_label'].value_counts().to_dict()

class_weights = np.zeros(num_classes)
for label in range(num_classes):
    if label in counts_dict and counts_dict[label] > 0:
        class_weights[label] = 1.0 / counts_dict[label]
    else:
        class_weights[label] = 0.0

sample_weights = [class_weights[label] for label in train_df['encoded_label']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

BATCH_SIZE = 256

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 5. MODEL DEFINITION (FULL BACKBONE FINE-TUNING)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 Target Cluster Hardware Core: {device}")

model = vgg16(weights=VGG16_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# ==========================================
# 6. PIPELINE TRAINING LOOP WITH HISTORY TRACKING
# ==========================================
EPOCHS = 15
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 3  

# Arrays to capture data for the Loss History Chart
train_losses = []
val_losses = []

print("\n🏋️ Starting full-scale deep fine-tuning loop on L40 GPU...")

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * images.size(0)
        
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Step Loss: {loss.item():.4f}")
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation Execution
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    print(f"🏁 Finished Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    scheduler.step(epoch_val_loss)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_vgg16_wafer_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Halting optimizations.")
            break

# ==========================================
# GENERATE CHART 1: TRAINING LOSS HISTORY
# ==========================================
print("\n📊 Generating Training History Visualizations...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='#0ea5e9', label='Training Loss', linewidth=2)
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', color='#f43f5e', label='Validation Loss', linewidth=2)
plt.title('VGG16 Cross-Entropy Loss Optimization Trace', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, len(train_losses) + 1))
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig('training_metrics_history.png', dpi=300)
plt.close()
print("💾 Saved: training_metrics_history.png")

# ==========================================
# 7. METRIC EVALUATION & CONFUSION MATRIX CHART
# ==========================================
print("\nLoading optimal saved metrics weights state for testing evaluations...")
model.load_state_dict(torch.load('best_vgg16_wafer_model.pth'))
model.eval()

all_preds = []
all_targets = []

print("Running testing inferences across all unseen validation points...")
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

unique_test_classes = np.unique(all_targets)
present_class_names = label_encoder.inverse_transform(unique_test_classes)

# Extract diagnostic dictionaries for automated chart graphing
report_dict = classification_report(all_targets, all_preds, labels=unique_test_classes, target_names=present_class_names, output_dict=True)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_targets, all_preds, labels=unique_test_classes, target_names=present_class_names))

# Generate Raw Confusion Matrix
cm = confusion_matrix(all_targets, all_preds, labels=unique_test_classes)
print("=== CONFUSION MATRIX ===")
print(cm)

# ==========================================
# GENERATE CHART 2: VISUAL CONFUSION MATRIX HEATMAP
# ==========================================
print("\n🎨 Generating Confusion Matrix Heatmap Chart...")
plt.figure(figsize=(11, 9))
sns.set_theme(style='white')

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=present_class_names, 
    yticklabels=present_class_names,
    cbar_kws={'label': 'Wafer Inspection Count'},
    annot_kws={'size': 11, 'weight': 'bold'}
)

plt.title('Final Unbiased VGG16 Silicon Defect Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Defect Classification Target', fontsize=12, labelpad=10)
plt.ylabel('True Factory Expert Defect Label', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=300)
plt.close()
print("💾 Saved: final_confusion_matrix.png")

# ==========================================
# NEW GENERATED CHART 4: PERFORMANCE METRICS VISUALIZATION (PRECISION, RECALL, F1, ACCURACY)
# ==========================================
print("\n📊 Generating Complete Performance Metrics Grouped Bar Chart...")
classes_list = [c for c in present_class_names]
precisions = [report_dict[c]['precision'] for c in classes_list]
recalls = [report_dict[c]['recall'] for c in classes_list]
f1_scores = [report_dict[c]['f1-score'] for c in classes_list]
global_accuracy = report_dict['accuracy']

x_axis_indices = np.arange(len(classes_list))
bar_width = 0.25

plt.figure(figsize=(14, 7))
sns.set_theme(style='whitegrid')

# Draw grouped bars for detailed class checks
plt.bar(x_axis_indices - bar_width, precisions, width=bar_width, label='Precision', color='#0ea5e9')
plt.bar(x_axis_indices, recalls, width=bar_width, label='Recall', color='#10b981')
plt.bar(x_axis_indices + bar_width, f1_scores, width=bar_width, label='F1-Score', color='#f43f5e')

# Reference marker overlay indicating global aggregate accuracy boundary
plt.axhline(y=global_accuracy, color='#64748b', linestyle='--', linewidth=1.5, label=f'Global Model Accuracy ({global_accuracy*100:.2f}%)')

plt.title('VGG16 Classification Metrics Profile per Target Defect Category', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Silicon Wafer Defect Categories', fontsize=12, labelpad=12)
plt.ylabel('Performance Metrics Score (Range 0.0 - 1.0)', fontsize=12, labelpad=12)
plt.xticks(x_axis_indices, classes_list, rotation=35, ha='right', fontsize=11)
plt.ylim(0.0, 1.05)
plt.legend(loc='lower left', fontsize=11, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
plt.tight_layout()
plt.savefig('final_metrics_comparison.png', dpi=300)
plt.close()
print("💾 Saved: final_metrics_comparison.png")

# ==========================================
# GENERATE CHART 3: PIPELINE HYPERPARAMETER SUMMARY DOCUMENT DIAGRAM
# ==========================================
print("📝 Generating Hyperparameter Pipeline Summary Card...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

pipeline_text = (
    "===========================================================\n"
    "            SILICON WAFER DEFECT DETECTION PIPELINE INFRASTRUCTURE \n"
    "===========================================================\n\n"
    f"  • Hardware Accelerator Core : NVIDIA L40 (48GB VRAM Dedicated Cluster)\n"
    f"  • Full Dataset Target Scale : Total Processed: {len(df_filtered)} expert-verified records\n"
    f"  • Sub-Split Partitioning    : 90% Training | 5% Validation | 5% Testing (Stratified Split)\n"
    f"  • Input Grid Resolution     : 224 x 224 x 3 RGB Matrix (Upscaled via 0-Order Interpolation)\n"
    f"  • Pixel Normalization Range : [0.0, 1.0] Unified Float Interval\n"
    f"  • Imbalance Mitigation      : Batch-Level Weighted Random Sampler (1.0 / Class Frequency)\n"
    f"  • Mini-Batch Capacity       : {BATCH_SIZE} Wafers per Processing Step\n"
    f"  • Network Backbone Core     : Pretrained VGG16 Architecture (Full Weights Fine-Tuning Active)\n"
    "  • Custom Output Layer Head  : Dense Classifier (4096 Linear -> ReLU -> Dropout 0.5 -> 9 Classes)\n"
    f"  • Hard Target Execution Limit: {EPOCHS} Epochs Maximum Run Bound\n"
    "  • Optimization Strategy     : Adam Optimizer (Base Learning Rate = 1e-5, Low Smooth Factor)\n"
    "  • Loss Objective Function   : Multi-Class Cross-Entropy Loss\n"
    f"  • Learning Decay Scheduler  : ReduceLROnPlateau (Patience=2 Epochs, Decay Factor=0.1)\n"
    f"  • Overfitting Early Guard   : EarlyStopping (Patience={early_stopping_patience} Non-Improving Epochs)\n\n"
    "==========================================================="
)

ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes, fontsize=11, fontfamily='monospace', verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='#f8fafc', edgecolor='#cbd5e1'))
plt.tight_layout()
plt.savefig('preprocessing_and_parameters.png', dpi=300)
plt.close()
print("💾 Saved: preprocessing_and_parameters.png\n✅ Visualizations built successfully!")