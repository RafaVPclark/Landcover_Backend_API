# inference.py — versão aprimorada com base no script antigo

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib
from matplotlib.colors import ListedColormap

# =========================================================
# 1. Configurações
# =========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
MODEL_PATH = "best_model_50.pth"

# =========================================================
# 2. Carregar modelo treinado
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
# 3. Transformação da imagem
# =========================================================
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # igual da versão antiga
    ToTensorV2()
])

# =========================================================
# 4. Paleta e rótulos fixos
# =========================================================
label_names = {
    0: "Fundo",
    1: "Edifícios",
    2: "Estradas",
    3: "Água",
    4: "Floresta",
}

PALETTE = {
    0: (70, 130, 180),   # Fundo
    1: (255, 0, 0),      # Edifícios
    2: (139, 69, 19),    # Estradas
    3: (0, 191, 255),    # Água
    4: (34, 139, 34),    # Floresta
}

# =========================================================
# 5. Funções auxiliares
# =========================================================
def decode_segmap(mask):
    """Converte IDs de classes em cores RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color_mask[mask == cls_id] = rgb
    return color_mask

def remap_mask(mask, mapping):
    """Permite corrigir confusões de classes (ex: estrada ↔ floresta)."""
    out = mask.copy()
    for old, new in mapping.items():
        out[mask == old] = new
    return out

# =========================================================
# 6. Predição principal
# =========================================================
def predict_segmentation(img_np):
    """Recebe uma imagem RGB numpy e retorna imagem redimensionada e máscara predita."""
    image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.shape[-1] == 3 else img_np
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensiona e transforma
    original = cv2.resize(image, (512, 512))
    transformed = transform(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return original, pred

# =========================================================
# 7. Função principal da API
# =========================================================
def generate_overlay(pil_image):
    """Função principal compatível com a API (usa PIL e retorna PILs + stats)."""

    # Converter PIL → NumPy (RGB)
    img_np = np.array(pil_image.convert("RGB"))

    # Rodar predição
    image, mask = predict_segmentation(img_np)

    # 🔹 Opcional: corrigir confusão entre classes
    # (exemplo: se o modelo costuma trocar estrada (2) com floresta (4))
    mask = remap_mask(mask, {2: 4, 4: 2})

    # Gerar máscara colorida e overlay
    color_mask = decode_segmap(mask)
    overlay = (0.6 * image.astype(float) + 0.4 * color_mask.astype(float)).astype(np.uint8)

    # Estatísticas de classes
    unique_classes, counts = np.unique(mask, return_counts=True)
    stats = {
        label_names.get(cls, "Desconhecido"): int(cnt)
        for cls, cnt in zip(unique_classes, counts)
    }

    # Converter para PIL
    mask_img = Image.fromarray(color_mask)
    overlay_img = Image.fromarray(overlay)

    return mask_img, overlay_img, stats
