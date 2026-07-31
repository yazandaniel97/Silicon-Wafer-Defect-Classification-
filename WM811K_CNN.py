# ==============================================================
# WK811 Silicon Wafer Defect Classification — Full Pipeline
# ==============================================================
# SECTIONS:
#   1.  Imports & setup
#   2.  Load data
#   3.  Filter supervised rows
#   4.  Encode labels
#   5.  Preprocessing (numpy)
#   6.  Oversampling (imblearn + augmentation)
#   7.  Stratified split  (80 / 10 / 10)
#   8.  tf.data pipelines
#   9.  CNN model (from scratch)
#   10. Compile & train  (CHANGED: Updated to 100 epochs)
#   11. Evaluate on test set
#   12. Visualisations
#         12a. Loss curves
#         12b. Confusion matrix
#         12c. Per-class metrics bar chart (F1 + Accuracy)
#         12d. Per-class metrics table (%)
# ==============================================================


# --------------------------------------------------------------
# 1. Imports & setup
# --------------------------------------------------------------
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f'TF version : {tf.__version__}')
print(f'GPU        : {tf.config.list_physical_devices("GPU")}')


# --------------------------------------------------------------
# 2. Load data
# ── Change PKL_PATH to your actual file path on the server ──
# --------------------------------------------------------------
PKL_PATH    = 'WM811K.pkl'
RESULTS_DIR = './'                       # saves next to this script

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(PKL_PATH, 'rb') as f:
    df_raw = pickle.load(f)

print('Raw shape:', df_raw.shape)
print('Columns  :', df_raw.columns.tolist())
print(df_raw.head(3))

# --------------------------------------------------------------
# 3. Filter supervised rows (FIXED FOR MIXED TYPES)
# --------------------------------------------------------------
# Drop missing rows in trainTestLabel
df = df_raw[df_raw['trainTestLabel'].notna()].copy()

# Filter row-by-row strictly checking the label column as a string.
df = df[df['trainTestLabel'].apply(lambda x: str(x).strip() not in ['0', '0.0', '[]'])].copy()

def flatten_label(x):
    if isinstance(x, list):
        if len(x) > 0:
            inner = x[0]
            return inner[0] if isinstance(inner, list) else inner
        return None
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.flatten()[0]
    return x

# Extract the raw label
df['label'] = df['failureType'].apply(flatten_label)

# Convert all labels to clean strings right now to prevent mixed-type comparison errors
df['label'] = df['label'].astype(str).str.strip()

# Filter out BOTH 'none' and any variation of '0'
df = df[~df['label'].isin(['none', '0', '0.0', 'nan', ''])].copy()
df = df.reset_index(drop=True)

print(f'Supervised samples : {len(df):,}')
print(df['label'].value_counts())


# --------------------------------------------------------------
# 4. Encode labels
# --------------------------------------------------------------
CLASS_NAMES = sorted(df['label'].unique().tolist())
NUM_CLASSES = len(CLASS_NAMES)
label2idx   = {cls: i for i, cls in enumerate(CLASS_NAMES)}
idx2label   = {i: cls for cls, i in label2idx.items()}

df['label_idx'] = df['label'].map(label2idx)

print(f'NUM_CLASSES : {NUM_CLASSES}')
print(f'CLASS_NAMES : {CLASS_NAMES}')


# --------------------------------------------------------------
# 5. Preprocessing (numpy)
# --------------------------------------------------------------
TARGET_SIZE = (32, 32)

def preprocess_wafer_map(wafer_map):
    """
    1. Convert to float32
    2. Nearest-neighbour resize to 32x32 (preserves {0,1,2})
    3. Normalise {0,1,2} -> {0.0, 0.5, 1.0}
    4. Add channel dim -> (32, 32, 1)
    """
    arr          = np.array(wafer_map, dtype=np.float32)
    h_in, w_in   = arr.shape
    h_out, w_out = TARGET_SIZE
    row_idx      = (np.arange(h_out) * h_in / h_out).astype(int)
    col_idx      = (np.arange(w_out) * w_in / w_out).astype(int)
    resized      = arr[np.ix_(row_idx, col_idx)]
    normed       = resized / 2.0
    return normed[:, :, np.newaxis]

print('Preprocessing wafer maps...')
df['processed'] = df['waferMap'].apply(preprocess_wafer_map)
print('Done.')
print(f'Sample shape  : {df["processed"].iloc[0].shape}')
print(f'Unique pixels : {np.unique(df["processed"].iloc[0])}')


# --------------------------------------------------------------
# 6. Oversampling (imblearn + augmentation)
# --------------------------------------------------------------
def augment_wafer(arr):
    """
    Safe augmentations only — rotations + flips.
    arr: (32, 32, 1) numpy array
    """
    img = arr[:, :, 0].copy()
    img = np.rot90(img, np.random.randint(0, 4))
    if np.random.rand() > 0.5: img = np.fliplr(img)
    if np.random.rand() > 0.5: img = np.flipud(img)
    return img[:, :, np.newaxis].copy()

X_raw = np.stack(df['processed'].values)
y_raw = df['label_idx'].values
N     = X_raw.shape[0]

X_flat            = X_raw.reshape(N, -1)
ros               = RandomOverSampler(random_state=SEED)
X_res_flat, y_res = ros.fit_resample(X_flat, y_raw)

X_res = X_res_flat.reshape(-1, 32, 32, 1)

for i in range(N, len(X_res)):
    X_res[i] = augment_wafer(X_res[i])

print(f'Before oversampling : {N:,}')
print(f'After oversampling  : {len(X_res):,}')
print(f'Class counts        : {sorted(Counter(y_res).items())}')


# --------------------------------------------------------------
# 7. Stratified split  (80 / 10 / 10)
# --------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_res, y_res, test_size=0.20, stratify=y_res, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f'Train : {len(X_train):,}  (80%)')
print(f'Val   : {len(X_val):,}   (10%)')
print(f'Test  : {len(X_test):,}   (10%)')


# --------------------------------------------------------------
# 8. tf.data pipelines
# --------------------------------------------------------------
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

def augment_tf(x, y):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

def make_dataset(X, y, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, seed=SEED)
    if augment:
        ds = ds.map(augment_tf, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, shuffle=True, augment=True)
val_ds   = make_dataset(X_val,   y_val)
test_ds  = make_dataset(X_test,  y_test)

print(f'Train batches : {len(train_ds)}')
print(f'Val batches   : {len(val_ds)}')
print(f'Test batches  : {len(test_ds)}')


# --------------------------------------------------------------
# 9. CNN model (from scratch)
# --------------------------------------------------------------
def build_cnn(num_classes, input_shape=(32, 32, 1), dropout=0.4):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1 — 32x32 -> 16x16
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2 — 16x16 -> 8x8
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3 — 8x8 (no pool)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling -> 128-d vector
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_cnn(NUM_CLASSES)
model.summary()


# --------------------------------------------------------------
# 10. Compile & train  (CHANGED: Configured to 100 epochs)
# --------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ADJUSTED: Scaled up early stopping and scheduler patience steps for longer optimization runs
callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, 'wk811_cnn_best.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=6, min_lr=1e-5, verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

print('\nTraining complete.')
print(f'Best val accuracy : {max(history.history["val_accuracy"])*100:.2f}%')


# --------------------------------------------------------------
# 11. Evaluate on test set
# --------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f'Test loss     : {test_loss:.4f}')
print(f'Test accuracy : {test_acc*100:.2f}%')

y_probs = model.predict(test_ds, verbose=0)
y_preds = np.argmax(y_probs, axis=1)
y_true  = np.concatenate([y for _, y in test_ds])


# --------------------------------------------------------------
# 12a. Loss curves
# --------------------------------------------------------------
ep = range(1, len(history.history['loss']) + 1)
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ep, history.history['loss'],     label='Train', color='#378ADD', lw=2)
ax.plot(ep, history.history['val_loss'], label='Val',   color='#D85A30', lw=2)
ax.set_title('Loss curves', fontsize=13)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

best_ep = history.history['val_loss'].index(min(history.history['val_loss']))
gap     = (history.history['accuracy'][best_ep] -
           history.history['val_accuracy'][best_ep]) * 100
print(f'Best epoch    : {best_ep + 1}')
print(f'Best val acc  : {history.history["val_accuracy"][best_ep]*100:.2f}%')
print(f'Train-val gap : {gap:.2f}%  ', end='')
if   gap > 10: print('-> possible overfitting')
elif gap < 1:  print('-> possible underfitting')
else:          print('-> healthy')
print('Saved: loss_curves.png')


# --------------------------------------------------------------
# 12b. Confusion matrix
# --------------------------------------------------------------
cm      = confusion_matrix(y_true, y_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5, annot_kws={'size': 9})
axes[0].set_title('Confusion matrix — raw counts', fontsize=13)
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True',      fontsize=11)
axes[0].tick_params(axis='x', rotation=40)
axes[0].tick_params(axis='y', rotation=0)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5,
            vmin=0, vmax=1, annot_kws={'size': 9})
axes[1].set_title('Confusion matrix — normalised (recall)', fontsize=13)
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True',      fontsize=11)
axes[1].tick_params(axis='x', rotation=40)
axes[1].tick_params(axis='y', rotation=0)

plt.suptitle('Confusion matrix', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: confusion_matrix.png')


# --------------------------------------------------------------
# 12c. Per-class metrics bar chart — F1, Precision, Recall, Accuracy (%)
# --------------------------------------------------------------
report = classification_report(
    y_true, y_preds,
    target_names=CLASS_NAMES,
    output_dict=True
)

per_class_acc = cm.diagonal() / cm.sum(axis=1)

metrics_df = pd.DataFrame({
    'Precision' : [report[c]['precision'] * 100 for c in CLASS_NAMES],
    'Recall'    : [report[c]['recall']    * 100 for c in CLASS_NAMES],
    'F1'        : [report[c]['f1-score']  * 100 for c in CLASS_NAMES],
    'Accuracy'  : [per_class_acc[i]       * 100 for i in range(NUM_CLASSES)],
}, index=CLASS_NAMES)

x   = np.arange(NUM_CLASSES)
w   = 0.20
fig, ax = plt.subplots(figsize=(15, 6))

ax.bar(x - 1.5*w, metrics_df['Precision'], w, label='Precision', color='#378ADD', alpha=0.87)
ax.bar(x - 0.5*w, metrics_df['Recall'],    w, label='Recall',    color='#1D9E75', alpha=0.87)
ax.bar(x + 0.5*w, metrics_df['F1'],        w, label='F1-score',  color='#D85A30', alpha=0.87)
ax.bar(x + 1.5*w, metrics_df['Accuracy'],  w, label='Accuracy',  color='#8B5CF6', alpha=0.87)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=10)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)')
ax.set_title('Per-class Precision / Recall / F1 / Accuracy (%)', fontsize=13)
ax.axhline(80, color='gray', linestyle='--', lw=0.8, alpha=0.5, label='80% threshold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: per_class_metrics.png')


# --------------------------------------------------------------
# 12d. Per-class metrics table (%)
# --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

col_labels = ['Class', 'Precision %', 'Recall %', 'F1 %', 'Accuracy %']
cell_data  = []
for i, cls in enumerate(CLASS_NAMES):
    cell_data.append([
        cls,
        f"{metrics_df.loc[cls, 'Precision']:.1f}%",
        f"{metrics_df.loc[cls, 'Recall']:.1f}%",
        f"{metrics_df.loc[cls, 'F1']:.1f}%",
        f"{metrics_df.loc[cls, 'Accuracy']:.1f}%",
    ])

table = ax.table(
    cellText  = cell_data,
    colLabels = col_labels,
    cellLoc   = 'center',
    loc       = 'center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.9)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#378ADD')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Colour-code cells by score — green if >=80%, amber if >=60%, red if <60%
for i, cls in enumerate(CLASS_NAMES):
    for j, metric in enumerate(['Precision', 'Recall', 'F1', 'Accuracy']):
        val = metrics_df.loc[cls, metric]
        if   val >= 80: color = '#E6F4EA'   # light green
        elif val >= 60: color = '#FFF8E1'   # light amber
        else:           color = '#FDECEA'   # light red
        table[i+1, j+1].set_facecolor(color)
    table[i+1, 0].set_facecolor('#F5F5F5')

plt.title('Per-class metrics table (%)', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_class_metrics_table.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Saved: per_class_metrics_table.png')

print('\nPer-class metrics (%):\n')
print(metrics_df.round(1).to_string())


# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------
saved_files = [
    'loss_curves.png',
    'confusion_matrix.png',
    'per_class_metrics.png',
    'per_class_metrics_table.png',
    'wk811_cnn_best.keras',
]

print('\n' + '=' * 55)
print('ALL SAVED FILES')
print('=' * 55)
for fname in saved_files:
    path   = os.path.join(RESULTS_DIR, fname)
    status = '✓' if os.path.exists(path) else '✗ missing'
    print(f'  {status}  {fname}')
print('=' * 55)
print('Pipeline complete.')