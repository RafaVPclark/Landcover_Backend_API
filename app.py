# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from inference import generate_overlay

from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API DO CLARK - Segmentação de Imagem"}

@app.post("/api/generateImageMask")
async def generate_image_mask(file: UploadFile = File(...)):
    # Ler bytes da imagem enviada
    image_bytes = await file.read()
    
    # Abrir imagem em memória
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Rodar inferência (deve retornar mask e overlay em memória)
    mask_img, overlay_img, stats = generate_overlay(image)

    # Converter imagens para base64
    def img_to_base64(img):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    mask_b64 = img_to_base64(mask_img)
    overlay_b64 = img_to_base64(overlay_img)

    return JSONResponse({
        "mask_base64": mask_b64,
        "overlay_base64": overlay_b64,
        "class_stats": stats
    })
