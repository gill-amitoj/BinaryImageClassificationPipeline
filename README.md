# Intelligent Image Classification

A fun learning project — built this to explore transfer learning, Grad-CAM, and end-to-end ML pipelines. Not deployed anywhere; just a local experiment for learning purposes.

<img width="1353" height="756" alt="image" src="https://github.com/user-attachments/assets/51cfa61a-ccd7-4c50-bd21-b89e83eec343" />
<img width="1199" height="734" alt="image" src="https://github.com/user-attachments/assets/b9b2b64c-741f-45a5-861b-da1afb1d8b74" />

## What I Used
- **Python**, **PyTorch**, **torchvision**
- **ResNet-18** (pretrained on ImageNet) with fine-tuned last block
- **Grad-CAM** to visualize what the model focuses on
- **Flask** for a local web interface to test predictions
- **scikit-learn** for evaluation metrics

## How to Run It

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Put your images in `data/train/` and `data/val/` (one folder per class), then:

```bash
# Train
python3 -m src.train --train_dir data/train --val_dir data/val

# Evaluate + Grad-CAM
python3 -m src.evaluate --data_dir data/val

# Try it locally
python3 app/app.py
# Open http://127.0.0.1:5000
```

## What's in Here
- `src/` – training, evaluation, model, data loading, Grad-CAM
- `models/` – saved weights and class mapping
- `data/` – dataset (ImageFolder layout)
- `app/` – Flask app for local testing

## Note
This was built for learning — not production. The dataset is small (20 images per class), so results are more of a proof of concept than a robust classifier. Use images you have the right to use.