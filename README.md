# Intelligent Image Classification

<<<<<<< HEAD
A practical image classification pipeline using transfer learning (ResNet-18), with evaluation metrics and Grad-CAM visualizations for model transparency.
=======
Transfer learning with ResNet‑18 for multi‑class image classification, evaluation with Grad‑CAM, and a Flask web app for real‑time inference.
<img width="1353" height="756" alt="image" src="https://github.com/user-attachments/assets/51cfa61a-ccd7-4c50-bd21-b89e83eec343" />
<img width="1199" height="734" alt="image" src="https://github.com/user-attachments/assets/b9b2b64c-741f-45a5-861b-da1afb1d8b74" />


>>>>>>> 76c965f46c9679de397d8d903368f47b191ee7e1

## Project Structure
- `src/` – data loading, model, training, evaluation
- `models/` – saved model and class mapping
- `data/` – dataset (ImageFolder format)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Organize your dataset:
```
data/
  train/
    class_a/ ...
    class_b/ ...
  val/
    class_a/ ...
    class_b/ ...
```

## Training
```bash
python3 -m src.train --train_dir data/train --val_dir data/val \
  --epochs 10 --batch_size 32 --lr 1e-3 --img_size 224 \
  --output_dir models --checkpoint_name resnet18_best.pth
```
- Model: `models/resnet18_best.pth`
- Classes: `models/classes.json`

## Evaluation & Grad-CAM
```bash
python3 -m src.evaluate --data_dir data/val \
  --model_path models/resnet18_best.pth \
  --classes_path models/classes.json \
  --out_dir models/eval --num_cam 6
```
- Metrics: `models/eval/metrics.json`
- Confusion matrix: `models/eval/confusion_matrix.png`
- Grad-CAM overlays: `models/eval/grad_cam/*.jpg`

## Ethics & Privacy
- Use only images you have rights to.
- Grad-CAM aids interpretability, not a guarantee of fairness.

---

<<<<<<< HEAD
=======
Start the web app to upload an image and see the prediction + Grad‑CAM:

```bash
python3 app/app.py
```

## Notes

- ResNet‑18 is loaded with ImageNet weights via the modern `ResNet18_Weights.DEFAULT` API.
- Transforms use ImageNet mean/std to match the pretrained backbone.
- Grad‑CAM targets the last block (`model.layer4[-1]`).

## Ethical Use & Privacy

- Use images you own or that are licensed for reuse. Don’t include sensitive or personal data without consent.
- By default, uploads are saved under `app/static/uploads/` and Grad‑CAM overlays under `app/static/outputs/`.
	- You can delete them from the result page. For production, prefer short‑lived storage or auto‑deletion.
- Grad‑CAM is an interpretability aid, not a guarantee. Don’t overclaim causality.
- If deploying externally:
	- Enable HTTPS/TLS and access controls (see Basic Auth below).
	- Provide a short privacy notice and a contact for data removal requests.

## Accessing the App

Local development (recommended):

```bash
python3 app/app.py
# http://127.0.0.1:5000
```

Expose to your LAN (will prompt macOS local network permission):

```bash
HOST=0.0.0.0 PORT=8000 python3 app/app.py
# http://<your-lan-ip>:8000
```

Public sharing (temporary): use a tunnel like `ngrok` to your local port.

Production options:
- Containerize and deploy to Cloud Run, Fly.io, or Railway (TLS, scaling, secrets).
- Use a WSGI server (e.g., gunicorn) in front of Flask, with HTTPS termination via a reverse proxy (Caddy/Nginx) or a managed platform.
>>>>>>> 76c965f46c9679de397d8d903368f47b191ee7e1

