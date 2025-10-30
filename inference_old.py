# =========================================================
# Segmentação de Imagem com U-Net (visualização aprimorada)
# =========================================================

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib
from matplotlib.colors import ListedColormap

# =========================================================
# 1️ Configuração
# =========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Usando dispositivo:", DEVICE)

NUM_CLASSES = 5
MODEL_PATH = "best_model_50.pth"

# =========================================================
# 2️ Carregar modelo treinado
# =========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()

# =========================================================
# 3️ Transformação da imagem
# =========================================================
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# =========================================================
# 🟦 Função de ajuste de contraste e saturação
# =========================================================
def adjust_contrast_and_saturation(image, contrast=1.3, saturation=0.6):
    """
    Aumenta o contraste e reduz a saturação da imagem.
    contrast > 1 aumenta contraste, < 1 reduz.
    saturation < 1 reduz cores, > 1 aumenta.
    """
    # --- Ajuste de contraste ---
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    # --- Conversão para HSV para ajustar saturação ---
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation  # canal de saturação (S)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return adjusted

# =========================================================
# 4️ Função de predição
# =========================================================
def predict_segmentation(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Falha ao carregar imagem: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 🔹 Aplica o ajuste de contraste e saturação
    # adjusted_image = adjust_contrast_and_saturation(image, contrast=1.4, saturation=0.5)

    # 🔹 Mostra lado a lado antes e depois
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title("Antes do ajuste")
    axs[0].axis("off")

    axs[1].imshow(image)
    axs[1].set_title("Após ajuste (contraste ↑, saturação ↓)")
    axs[1].axis("off")
    plt.show()

    # 🔹 Redimensiona e transforma a imagem ajustada
    original = cv2.resize(image, (512, 512))
    transformed = transform(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return original, pred

# =========================================================
# 5️ Rótulos e paleta fixa (cores intuitivas)
# =========================================================
label_names = {
    0: "Fundo",
    1: "Edifícios",
    2: "Estradas",
    3: "Água",
    4: "Floresta",
}

PALETTE = {
    0: (70, 130, 180),   # Fundo - azul acinzentado
    1: (255, 0, 0),      # Edifícios - vermelho
    2: (139, 69, 19),    # Estradas - marrom
    3: (0, 191, 255),    # Água - azul claro
    4: (34, 139, 34),    # Floresta - verde
}

palette_rgb = [tuple(v/255.0 for v in PALETTE[i]) for i in range(NUM_CLASSES)]
listed_cmap = ListedColormap(palette_rgb)

def decode_segmap(mask):
    """Converte IDs de classes em cores RGB"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color_mask[mask == cls_id] = rgb
    return color_mask

# =========================================================
# 6️ Funções auxiliares
# =========================================================
def remap_mask(mask, mapping):
    """Remapeia IDs de classes"""
    out = mask.copy()
    for old, new in mapping.items():
        out[mask == old] = new
    return out

def per_class_iou(y_true, y_pred, num_classes=NUM_CLASSES):
    """Calcula IoU por classe"""
    ious = {}
    for cls in range(num_classes):
        intersection = np.logical_and(y_true == cls, y_pred == cls).sum()
        union = np.logical_or(y_true == cls, y_pred == cls).sum()
        iou = intersection / union if union > 0 else float("nan")
        ious[cls] = iou
    return ious

def show_label_image(label_mask, ax=None, title="Máscara (IDs)"):
    if ax is None:
        _, ax = plt.subplots(1,1)
    im = ax.imshow(label_mask, cmap=listed_cmap, vmin=0, vmax=NUM_CLASSES-1)
    ax.set_title(title)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(NUM_CLASSES))
    cbar.ax.set_yticklabels([f"{i}: {label_names[i]}" for i in range(NUM_CLASSES)])
    return ax

# =========================================================
# 7️ Visualização aprimorada
# =========================================================
def visualize_result_improved(image_path, gt_mask_path=None, class_swap_mapping=None):
    image, mask_pred = predict_segmentation(image_path)

    if class_swap_mapping:
        mask_pred = remap_mask(mask_pred, class_swap_mapping)

    color_mask = decode_segmap(mask_pred)
    overlay = (0.6 * image.astype(float) + 0.4 * color_mask.astype(float)).astype(np.uint8)

    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,0.6])
    ax_img = fig.add_subplot(gs[:,0])
    ax_label = fig.add_subplot(gs[0,1])
    ax_overlay = fig.add_subplot(gs[1,1])
    ax_legend = fig.add_subplot(gs[:,2])
    ax_legend.axis("off")

    ax_img.imshow(image); ax_img.set_title("Imagem original ajustada"); ax_img.axis("off")
    show_label_image(mask_pred, ax=ax_label, title="Máscara predita (IDs)")
    ax_overlay.imshow(overlay); ax_overlay.set_title("Overlay (imagem + máscara)"); ax_overlay.axis("off")

    patches = [matplotlib.patches.Patch(color=palette_rgb[i], label=f"{i}: {label_names[i]}") for i in range(NUM_CLASSES)]
    ax_legend.legend(handles=patches, loc='center')
    plt.tight_layout()

    unique_classes, counts = np.unique(mask_pred, return_counts=True)
    print("Classes detectadas (predição):")
    for cls, cnt in zip(unique_classes, counts):
        print(f"  {cls}: {label_names.get(cls,'Desconhecido')} — {cnt} pixels")

    plt.show()

# =========================================================
# 8️ Rodar
# =========================================================
image_path = "teste8.jpg"
visualize_result_improved(image_path, class_swap_mapping={2:4, 4:2})
