import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import torch
from torchvision import transforms
from sklearn.manifold import TSNE
import torch.nn.functional as F
from tqdm import tqdm
from IPython.display import HTML
import matplotlib.animation as animation
import pandas as pd
from data.action_units_prompts import AU_PROMPTS, CLASS_PROMPTS
from clip.custom_clip import TemporalGradCAM, patch_clip_attention

import re
import os

def get_effective_weights(model):
    clf = model.temporal_classifier
    if not isinstance(clf, torch.nn.Sequential) or len(clf) < 3:
        raise ValueError("Expected a 2-layer MLP classifier (Linear → ReLU → Linear)")

    W1 = clf[0].weight.detach().cpu()      # [H, N]
    W2 = clf[2].weight.detach().cpu()      # [C, H]
    W_eff = W2 @ W1                        # [C, N]
    return W_eff.numpy()

def plot_pos_au_per_class_n_diff(model, au_prompts, classnames):

    """
    Show top unique AUs most important for each class,
    and highlight those that are common across multiple classes.
    """

    os.makedirs("visual", exist_ok=True)

    # ✅ Find the last Linear layer
    classifier = model.temporal_classifier
    last_linear = None
    if isinstance(classifier, torch.nn.Sequential):
        for layer in reversed(classifier):
            if isinstance(layer, torch.nn.Linear):
                last_linear = layer
                break
    elif isinstance(classifier, torch.nn.Linear):
        last_linear = classifier
    if last_linear is None:
        raise AttributeError("No nn.Linear layer found inside temporal_classifier")

    # ✅ Extract weights
    weights = last_linear.weight.detach().cpu().numpy()
    if weights.shape[0] != len(classnames) and weights.shape[1] == len(classnames):
        weights = weights.T
    num_classes, num_aus = weights.shape

    print(f"Weight matrix shape: {weights.shape}")

    # ✅ Trim or pad AU prompts to match shape
    if num_aus != len(au_prompts):
        print(f"[Warning] Adjusting AU prompts from {len(au_prompts)} → {num_aus}")
        au_prompts = au_prompts[:num_aus]

    # ✅ Identify positive AUs for each class
    positive_aus_per_class = {}
    for class_idx, emotion in enumerate(classnames):
        class_weights = weights[class_idx]
        min_len = min(len(class_weights), len(au_prompts))
        class_weights = class_weights[:min_len]
        aus_trimmed = au_prompts[:min_len]

        mask = class_weights > 0
        pos_aus = np.array(aus_trimmed)[mask]
        positive_aus_per_class[emotion] = list(pos_aus)

    # ✅ Find AUs common to more than one class
    all_positive_aus = sum(positive_aus_per_class.values(), [])
    au_counts = {au: all_positive_aus.count(au) for au in set(all_positive_aus)}
    common_aus = [au for au, count in au_counts.items() if count > 1]
    print(f"Common AUs across classes: {common_aus}")

    # 🎨 Plot per class
    for class_idx, emotion in enumerate(classnames):
        if class_idx >= weights.shape[0]:
            break
        class_weights = weights[class_idx]
        min_len = min(len(class_weights), len(au_prompts))
        class_weights = class_weights[:min_len]
        aus_trimmed = au_prompts[:min_len]

        mask = class_weights > 0
        pos_vals = class_weights[mask]
        pos_aus = np.array(aus_trimmed)[mask]

        if len(pos_aus) == 0:
            print(f"No positive-weight AUs for class '{emotion}'")
            continue

        sorted_idx = np.argsort(pos_vals)[::-1]
        pos_vals = pos_vals[sorted_idx]
        pos_aus = pos_aus[sorted_idx]

        # ✅ Dynamic figure size
        fig_height = max(3, 0.4 * len(pos_aus))
        plt.figure(figsize=(6, fig_height))

        # ✅ Color-code bars: common AUs vs unique
        colors = ["gold" if au in common_aus else "tomato" for au in pos_aus]

        bar_positions = np.arange(len(pos_aus))
        plt.barh(bar_positions, pos_vals, color=colors, height=0.6)
        plt.yticks(bar_positions, pos_aus)
        plt.title(f"Positive AUs for {emotion}")
        plt.xlabel("Importance Weight")
        plt.gca().invert_yaxis()
        plt.margins(y=0.1)
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.1)

        # ✅ Add legend once
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="tomato", label="Unique AUs"),
            plt.Rectangle((0, 0), 1, 1, color="gold", label="Common AUs"),
        ]
        plt.legend(handles=handles, loc="lower right", frameon=True)

        save_path = f"visual/positive_aus_with_diff_{emotion}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()


def plot_positive_aus_per_class(model, au_prompts, classnames, top_k=4, normalize=True):
    """
    Plot top-k AUs with positive weights for each class.
    Supports Linear or 2-layer MLP classifier (uses effective weights W2 @ W1).
    Normalizes by range (dividing by max) without shifting to 0.
    """

    os.makedirs("visual", exist_ok=True)

    # ✅ 1. Find linear layers
    classifier = model.temporal_classifier
    linear_layers = []
    if isinstance(classifier, torch.nn.Sequential):
        for layer in classifier:
            if isinstance(layer, torch.nn.Linear):
                linear_layers.append(layer)
    elif isinstance(classifier, torch.nn.Linear):
        linear_layers.append(classifier)

    if not linear_layers:
        raise AttributeError("No nn.Linear layer found inside temporal_classifier")

    # ✅ 2. Use effective weights if it's a two-layer MLP
    if len(linear_layers) >= 2:
        W1 = linear_layers[0].weight.detach().cpu()  # [H, N]
        W2 = linear_layers[-1].weight.detach().cpu() # [C, H]
        weights = (W2 @ W1).numpy()                  # [C, N]
        print(f"Using effective weights from 2-layer MLP: {weights.shape}")
    else:
        weights = linear_layers[0].weight.detach().cpu().numpy()
        print(f"Using direct linear layer weights: {weights.shape}")

    # ✅ 3. Ensure correct orientation [num_classes, num_AUs]
    if weights.shape[0] != len(classnames) and weights.shape[1] == len(classnames):
        weights = weights.T
    print(f"Adjusted weight shape: {weights.shape}")

    num_classes, num_aus = weights.shape
    if num_aus != len(au_prompts):
        print(f"[Warning] Adjusting AU prompts from {len(au_prompts)} → {num_aus}")
        au_prompts = au_prompts[:num_aus]

    # 🎨 4. Plot for each class
    for class_idx, emotion in enumerate(classnames):
        if class_idx >= weights.shape[0]:
            break

        class_weights = weights[class_idx]

        # Select only positive weights
        positive_mask = class_weights > 0
        pos_vals = class_weights[positive_mask]
        pos_aus = np.array(au_prompts)[positive_mask]

        if len(pos_aus) == 0:
            print(f"No positive-weight AUs for class '{emotion}'")
            continue

        # Sort descending
        sorted_idx = np.argsort(pos_vals)[::-1]
        pos_vals = pos_vals[sorted_idx]
        pos_aus = pos_aus[sorted_idx]

        # Select top-k
        if top_k > 0:
            pos_vals = pos_vals[:top_k]
            pos_aus = pos_aus[:top_k]

        # ✅ Normalize by max (range normalization)
        if normalize and len(pos_vals) > 0:
            max_val = np.max(np.abs(pos_vals))
            if max_val > 0:
                pos_vals = pos_vals / max_val  # scales to [min/max ratio]
            # no subtraction of min → preserves relative magnitude

        # Plot
        fig_height = max(3, 0.4 * len(pos_aus))
        plt.figure(figsize=(6, fig_height))
        bar_positions = np.arange(len(pos_aus))
        plt.barh(bar_positions, pos_vals, color="tomato", height=0.6)
        plt.yticks(bar_positions, pos_aus)
        plt.title(f"Top Positive AUs for {emotion}")
        plt.xlabel("Importance Weight" if normalize else "Raw Importance")
        plt.xlim(0, 1.05 if normalize else None)
        plt.gca().invert_yaxis()
        plt.margins(y=0.1)
        plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)

        save_path = f"visual/top{top_k}_positive_aus_{emotion}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.show()



def plot_au_most_import_per_class(model, au_prompts, classnames):
    """
    Show top unique AUs most important for each class.
    Works for both Linear and Sequential classifiers.
    """
    os.makedirs("visual", exist_ok=True)

    # ✅ Find the last Linear layer inside model.temporal_classifier
    classifier = model.temporal_classifier
    last_linear = None
    if isinstance(classifier, torch.nn.Sequential):
        for layer in reversed(classifier):
            if isinstance(layer, torch.nn.Linear):
                last_linear = layer
                break
    elif isinstance(classifier, torch.nn.Linear):
        last_linear = classifier
    if last_linear is None:
        raise AttributeError("No nn.Linear layer found inside temporal_classifier")

    # Extract weights
    weights = last_linear.weight.detach().cpu().numpy()

    # ✅ Ensure orientation is [num_classes, num_AUs]
    if weights.shape[0] != len(classnames):
        weights = weights.T
    print(f"Weight matrix shape: {weights.shape}")

    # 1️⃣ Assign each AU to the emotion where it’s most influential
    abs_weights = np.abs(weights)
    au_to_class = np.argmax(abs_weights, axis=0)   # [num_AUs] → class index

    # 2️⃣ Build mapping: emotion → its uniquely assigned AUs
    mapping = {emotion: [] for emotion in classnames}
    num_classes, num_aus = weights.shape
    if num_aus != len(au_prompts):
        print(f"[Warning] Adjusting AU prompts from {len(au_prompts)} → {num_aus}")
        au_prompts = au_prompts[:num_aus]

    for au_idx, class_idx in enumerate(au_to_class):
        if au_idx >= len(au_prompts):
            break  # stop if safety bound reached
        if class_idx < len(classnames):
            mapping[classnames[class_idx]].append(au_prompts[au_idx])

    # 3️⃣ For each emotion, rank its assigned AUs by strength
    for emotion, aus in mapping.items():
        class_idx = classnames.index(emotion)
        aus_sorted = sorted(
            aus,
            key=lambda x: abs(weights[class_idx, au_prompts.index(x)]),
            reverse=True
        )
        mapping[emotion] = aus_sorted[:min(30, len(aus_sorted))]

    # 4️⃣ Plot per class
    for i, emotion in enumerate(classnames):
        aus = mapping[emotion]
        if not aus:
            print(f"No dominant AUs for class {emotion}")
            continue

        vals = [weights[i, au_prompts.index(a)] for a in aus]
        plt.figure(figsize=(10, 3))
        plt.barh(aus, vals, color='tomato')
        plt.title(f"Unique Top AUs for {emotion}")
        plt.xlabel("Importance Weight")
        plt.gca().invert_yaxis()

        save_path = f"visual/au_class_heatmap_{emotion}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {save_path}")

        plt.show()


def plot_average_au_by_class(model, data_loader, au_prompts, classnames, gpu, device):
    sims, labels = get_au_similarities_from_loader(model, data_loader, au_prompts, gpu, device)

    # average AU by class
    means = []
    for c in range(len(classnames)):
        means.append(sims[labels==c].mean(dim=0))
    means = torch.stack(means).numpy()

    plt.figure(figsize=(12,6))
    sns.heatmap(means, xticklabels=au_prompts, yticklabels=classnames, cmap='viridis')
    
    plt.title("Average AU similarity per class")
    
    save_path = "visual/average_au_by_class.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Average AU by class saved to {save_path}")
    
    plt.show()


def plot_tsne_au_vectors(model, data_loader, au_prompts, gpu, device):
    """
    Compute AU similarity vectors for all samples and plot TSNE projection.
    """
    sims, labels = get_au_similarities_from_loader(model, data_loader, au_prompts, gpu, device)

    X = sims.numpy()
    y = labels.numpy()

    # you can filter to only two classes if you want (pain vs neutral):
    # mask = np.isin(y, [pain_class_id, neutral_class_id])
    # X = X[mask]
    # y = y[mask]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', alpha=0.7)

    # Add a legend with class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, np.unique(y), title="Classes")

    plt.title("t-SNE of AU similarity vectors (coloured by class)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    save_path = "visual/tsne_au_vectors.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"TSNE saved to {save_path}")

    plt.show()


def tsne_on_aus(model, data_loader, au_prompts, classnames, gpu, device):
    # 1. Get AU similarity vectors for all samples
    sims, labels = get_au_similarities_from_loader(model, data_loader, au_prompts, gpu, device)
    sims = sims.numpy()  # [N,num_AUs]
    labels = labels.numpy()

    num_classes = len(classnames)
    num_aus = sims.shape[1]

    # 2. Compute mean AU value per class
    class_means = []
    for c in range(num_classes):
        class_means.append(sims[labels==c].mean(axis=0))
    class_means = np.stack(class_means)  # [num_classes,num_AUs]

    # 3. Build feature vector for each AU across classes
    # each row = AU, each col = class mean
    aus_features = class_means.T  # [num_AUs,num_classes]
    perplexity = min(30, (num_aus - 1) // 3)  # safe rule of thumb

    # 4. Run t-SNE on AUs
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(aus_features)

    # 5. Assign color to each AU based on max class value
    max_class_idx = np.argmax(aus_features, axis=1)  # class with highest mean for each AU

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=max_class_idx, cmap='tab10', s=80)
    for i, txt in enumerate(au_prompts):
        plt.annotate(txt, (X_2d[i,0], X_2d[i,1]), fontsize=8)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, classnames, title="Class with highest AU mean")
    plt.title("t-SNE of AUs (each point = one AU)")

    # build relative features
    class_means = np.stack(class_means)  # [num_classes,num_aus]
    # standardise per-AU across classes
    normed = (class_means - class_means.mean(axis=0)) / (class_means.std(axis=0)+1e-6)
    aus_features = normed.T  # [num_aus,num_classes]

    # choose perplexity
    perplexity = min(30, max(2,(aus_features.shape[0]-1)//3))
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(aus_features)

    # color by class with highest relative activation
    max_class_idx = np.argmax(aus_features, axis=1)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=max_class_idx, cmap='tab10', s=80)
    for i, txt in enumerate(au_prompts):
        plt.annotate(txt, (X_2d[i,0], X_2d[i,1]), fontsize=8)
    handles,_ = scatter.legend_elements()
    plt.legend(handles, classnames, title="Class where AU is strongest (relative)")
    plt.title("t-SNE of AUs (relative strength)")

    save_path = "visual/tsne_au_vectors.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"TSNE saved to {save_path}")
    plt.show()


def get_au_similarities_from_loader(model, data_loader, au_prompts, gpu, device):
    """
    Loop over a DataLoader and collect AU similarity vectors and labels.
    """
    model.eval()
    all_sims = []
    all_labels = []

    iterable = data_loader
    iterable = tqdm(data_loader, desc="Extracting AU similarities")

    with torch.no_grad():
        for images, labels in iterable:
            images = images[0].cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            sims = model.compute_au_similarities(images, au_prompts, device)
            all_sims.append(sims.cpu())
            all_labels.append(labels.cpu())
    all_sims = torch.cat(all_sims)
    all_labels = torch.cat(all_labels)
    return all_sims, all_labels

def visualize_metrics_from_csv(csv_path="metrics/metrics_log.csv", save_dir="metrics/plots/transformers"):
    # 1️⃣ Load the data
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df.head())

    # Ensure correct types
    df["epoch"] = pd.to_numeric(df["epoch"])
    metrics = ["au_sim_mean", "au_sim_std", "logits_mean", "logits_std"]

    # 2️⃣ ---- LINE PLOTS ----
    plt.figure(figsize=(12, 8))
    for i, m in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.lineplot(data=df, x="epoch", y=m, hue="model", marker="o", linewidth=2)
        plt.title(m.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.grid(True)
    plt.tight_layout()

    line_plot_path = os.path.join(save_dir, "metrics_line_plots.png")
    plt.savefig(line_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Line plots saved to {line_plot_path}")
    plt.show()

    # 3️⃣ ---- HEATMAPS ----
    for m in metrics:
        # 🧩 Aggregate duplicates
        df_agg = df.groupby(["model", "epoch"], as_index=False).mean()

        pivot = df_agg.pivot(index="model", columns="epoch", values=m)
        plt.figure(figsize=(10, 4))
        sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".3f",
                    cbar_kws={"label": m}, linewidths=0.3)
        plt.title(f"{m.replace('_', ' ').title()} (Model vs Epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("Model")

        heatmap_path = os.path.join(save_dir, f"heatmap_{m}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        print(f"✅ Heatmap saved to {heatmap_path}")
        plt.show()


    print("🎨 Visualization complete!")

AU_REGIONS = {
    # 👁 Brow / Forehead region
    1:  (0.30, 0.12, 0.40, 0.10),  # Inner brow raiser
    2:  (0.25, 0.10, 0.50, 0.10),  # Outer brow raiser
    4:  (0.30, 0.18, 0.40, 0.10),  # Brow lowerer
    5:  (0.35, 0.15, 0.30, 0.10),  # Upper lid raiser

    # 👀 Eye region
    6:  (0.25, 0.28, 0.50, 0.18),  # Cheek raiser
    7:  (0.25, 0.28, 0.50, 0.18),  # Lid tightener
    43: (0.25, 0.28, 0.50, 0.18),  # Eyes closed
    44: (0.25, 0.28, 0.50, 0.18),  # Squint
    45: (0.25, 0.28, 0.50, 0.18),  # Blink
    46: (0.25, 0.28, 0.50, 0.18),  # Wink

    # 👃 Nose / Mid-face region
    9:  (0.40, 0.40, 0.20, 0.10),  # Nose wrinkler
    10: (0.38, 0.42, 0.24, 0.10),  # Upper lip raiser
    11: (0.38, 0.42, 0.24, 0.10),  # Nasolabial deepener

    # 😊 Cheeks / mid-face area
    12: (0.35, 0.52, 0.30, 0.15),  # Lip corner puller
    13: (0.35, 0.52, 0.30, 0.15),  # Cheek puffer
    14: (0.35, 0.52, 0.30, 0.15),  # Dimpler
    15: (0.35, 0.52, 0.30, 0.15),  # Lip corner depressor
    16: (0.35, 0.55, 0.30, 0.12),  # Lower lip depressor

    # 👄 Mouth / lips
    17: (0.38, 0.58, 0.24, 0.12),  # Chin raiser
    18: (0.35, 0.58, 0.30, 0.12),  # Lip pucker
    20: (0.33, 0.55, 0.34, 0.12),  # Lip stretcher
    22: (0.35, 0.55, 0.30, 0.12),  # Lip funneler
    23: (0.35, 0.60, 0.30, 0.10),  # Lip tightener
    24: (0.35, 0.60, 0.30, 0.10),  # Lip pressor
    25: (0.36, 0.64, 0.28, 0.10),  # Lips parted
    26: (0.35, 0.68, 0.30, 0.12),  # Jaw drop
    27: (0.35, 0.68, 0.30, 0.12),  # Mouth stretch
    28: (0.35, 0.64, 0.30, 0.10),  # Lip suck
    29: (0.35, 0.66, 0.30, 0.10),  # Jaw thrust
    30: (0.35, 0.50, 0.30, 0.12),  # Jaw sideways
    31: (0.35, 0.66, 0.30, 0.10),  # Jaw clench
    32: (0.35, 0.66, 0.30, 0.10),  # Lip bite
    33: (0.35, 0.64, 0.30, 0.10),  # Blow
    34: (0.35, 0.64, 0.30, 0.10),  # Puff
    35: (0.35, 0.64, 0.30, 0.10),  # Suck
    36: (0.35, 0.64, 0.30, 0.10),  # Tongue out

    # 🧏 Head / movement-related
    51: (0.35, 0.10, 0.30, 0.15),  # Head turn left
    52: (0.35, 0.10, 0.30, 0.15),  # Head turn right
    53: (0.35, 0.15, 0.30, 0.10),  # Head up
    54: (0.35, 0.20, 0.30, 0.10),  # Head down
    55: (0.35, 0.20, 0.30, 0.10),  # Head tilt left
    56: (0.35, 0.20, 0.30, 0.10),  # Head tilt right

    # 😬 Eye direction / movement
    61: (0.25, 0.25, 0.50, 0.10),  # Eyes turn left
    62: (0.25, 0.25, 0.50, 0.10),  # Eyes turn right
    63: (0.25, 0.20, 0.50, 0.10),  # Eyes up
    64: (0.25, 0.28, 0.50, 0.10),  # Eyes down
}


def detect_face_bbox(frame):
    """
    Detect a face bounding box using OpenCV's Haar cascade.
    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # fallback: assume central face covering 60% of height
        h, w = frame.shape[:2]
        # x1, y1, x2, y2 = int(0.2*w), int(0.1*h), int(0.8*w), int(0.9*h)
        x1, y1, x2, y2 = int(0.1*w), int(0.1*h), int(1.0*w), int(1.0*h)
        return x1, y1, x2, y2
    else:
        x, y, w, h = faces[0]
        return x, y, x + w, y + h


def scale_au_box_within_face(face_bbox, x, y, w, h):
    """
    Scale AU region inside the detected face bounding box.
    (x, y, w, h) are normalized [0–1] within the face.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    face_w, face_h = fx2 - fx1, fy2 - fy1
    x1 = int(fx1 + x * face_w)
    y1 = int(fy1 + y * face_h)
    x2 = int(fx1 + (x + w) * face_w)
    y2 = int(fy1 + (y + h) * face_h)
    return x1, y1, x2, y2


def overlay_au_regions_face_adapt(frame, top_aus, scores, itera, alpha=0.45):
    """
    Overlay AU activation regions, scaled to the actual detected face area.
    """
    img_h, img_w, _ = frame.shape
    overlay = frame.copy()
    cmap = plt.get_cmap("rainbow")

    # detect face first
    face_bbox = detect_face_bbox(frame)
    fx1, fy1, fx2, fy2 = face_bbox
    cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (255, 255, 255), 2)

    for i, au_name in enumerate(top_aus):
        try:
            au_id = int(''.join([c for c in au_name if c.isdigit()]))
        except:
            continue
        if au_id not in AU_REGIONS:
            continue

        # map AU region inside face bbox
        x, y, w, h = AU_REGIONS[au_id]
        x1, y1, x2, y2 = scale_au_box_within_face(face_bbox, x, y, w, h)

        # color by importance
        color = np.array(cmap(scores[i]))[:3] * 255
        color = tuple(map(int, color))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(overlay, au_name, (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    plt.figure(figsize=(8,8))
    plt.imshow(vis)
    plt.axis("off")
    plt.title("AU Importance Overlay (Face-Aligned)")

    save_path = "visual/grad_cam_au"+str(itera)+".png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grad saved to {save_path}")

    plt.show()
