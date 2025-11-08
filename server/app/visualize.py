import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import io, base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image_path: str):
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Resize to match model input (224x224)
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Choose target layer (last conv layer in ResNet)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Forward pass for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    # Run Grad-CAM for the predicted class
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])
    grayscale_cam = grayscale_cam[0, :]

    # Overlay heatmap (both images are now 224x224)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert to base64
    buf = io.BytesIO()
    plt.imsave(buf, visualization)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
