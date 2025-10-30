# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os
from inference import generate_overlay

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API DO CLARK - Segmentação de Imagem"}

@app.post("/api/generateImageMask")
async def generate_image_mask(file: UploadFile = File(...)):
    # Salvar a imagem recebida
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_path = os.path.join(upload_dir, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Rodar inferência
    mask_path, overlay_path, stats = generate_overlay(image_path)

    # Retornar resultados
    return JSONResponse({
        "mask_url": f"/{mask_path}",
        "overlay_url": f"/{overlay_path}",
        "class_stats": stats
    })

# Para servir arquivos estáticos
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
