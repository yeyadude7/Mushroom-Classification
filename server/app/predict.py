# app/predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import time, math, json, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load model dynamically
def load_model(model_path: str, classes_path: str):
    with open(classes_path, "r") as f:
        class_mapping = json.load(f)
    classes = [v for _, v in sorted(class_mapping.items(), key=lambda x: int(x[0]))]

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"✅ Model loaded with classes: {classes}")
    return model, classes

# ✅ Inference
def predict(model, image_path: str, classes):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
    end_time = time.time()

    conf, pred = torch.max(probs, 1)
    probs_list = probs.cpu().numpy()[0].tolist()

    entropy = -sum(p * math.log(p + 1e-9) for p in probs_list)
    sorted_probs = sorted(probs_list, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]

    result = {
        "class": classes[pred.item()],
        "confidence": round(conf.item(), 4),
        "probabilities": {cls: round(p, 4) for cls, p in zip(classes, probs_list)},
        "entropy": round(entropy, 6),
        "margin": round(margin, 6),
        "inference_time_ms": round((end_time - start_time) * 1000, 2),
    }
    if result["confidence"] < 0.6:
        result["note"] = "Low confidence — image may be ambiguous or unclear."
    return result
