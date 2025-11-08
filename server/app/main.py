# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.predict import load_model, predict
from app.visualize import generate_gradcam
from tempfile import NamedTemporaryFile
import os

app = FastAPI(title="Mushroom Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Updated paths and loader
MODEL_PATH = "models/mushroom_resnet50_finetuned.pth"
CLASSES_PATH = "models/classes.json"
model, classes = load_model(MODEL_PATH, CLASSES_PATH)

@app.get("/")
def root():
    return {"message": "Mushroom Classification API is running."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict(model, tmp_path, classes)
        return result
    finally:
        os.remove(tmp_path)

@app.post("/visualize")
async def visualize_image(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        cam_image_b64 = generate_gradcam(model, tmp_path)
        return {"gradcam_image": cam_image_b64}
    finally:
        os.remove(tmp_path)
