# 🍄 Mushroom Classifier

A full-stack mushroom classification web app built with **FastAPI** (Python) and **React + Bootstrap** frontend.

### Features
- Upload mushroom images to classify as *edible* or *poisonous*
- Visualize model attention with **Grad-CAM**
- FastAPI backend using ResNet-50 fine-tuned on merged datasets

### Stack
- **Backend:** FastAPI, PyTorch, torchvision
- **Frontend:** React, Bootstrap 5
- **Deployment:** DigitalOcean droplet + Nginx reverse proxy

### Usage
```bash
# Clone repository
git clone https://github.com/<your-username>/mushroom-classifier.git
cd mushroom-classifier

# Backend setup
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup
cd ../frontend
npm install
npm start
