# inference.py

import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
MODEL_PATH = "best_model_50.pth"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

PALETTE = {
    0: (70, 130, 180),
    1: (255, 0, 0),
    2: (139, 69, 19),
    3: (0, 191, 255),
    4: (34, 139, 34),
}

label_names = {
    0: "Fundo",
    1: "Edifícios",
    2: "Estradas",
    3: "Água",
    4: "Floresta",
}


def predict_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = cv2.resize(image, (512, 512))
    transformed = transform(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return original, pred


def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color_mask[mask == cls_id] = rgb
    return color_mask


def generate_overlay(image_path):
    image, mask = predict_segmentation(image_path)
    color_mask = decode_segmap(mask)
    overlay = (0.6 * image + 0.4 * color_mask).astype(np.uint8)

    mask_path = "uploads/mask.png"
    overlay_path = "uploads/overlay.png"

    cv2.imwrite(mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Estatísticas das classes
    unique_classes, counts = np.unique(mask, return_counts=True)
    stats = {
        label_names.get(cls, "Desconhecido"): int(cnt)
        for cls, cnt in zip(unique_classes, counts)
    }

    return mask_path, overlay_path, stats
