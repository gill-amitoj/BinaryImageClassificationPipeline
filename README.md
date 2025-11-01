# Intelligent Image Classification

Transfer learning with ResNet‑18 for multi‑class image classification, evaluation with Grad‑CAM, and a Flask web app for real‑time inference.

## Structure

- `src/` – dataset, model builder, training and evaluation scripts
- `models/` – saved checkpoints and class mapping
- `data/` – dataset in ImageFolder layout (`train/`, `val/`, optional `test/`)
- `app/` – Flask app with HTML templates and static assets

## Setup

Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Organize your dataset as:

```
data/
	train/
		class_a/ ...
		class_b/ ...
	val/
		class_a/ ...
		class_b/ ...
```

## Train

Trains the head of ResNet‑18 and saves the best checkpoint and class mapping:

```bash
python -m src.train --train_dir data/train --val_dir data/val \
	--epochs 10 --batch_size 32 --lr 1e-3 --img_size 224 \
	--output_dir models --checkpoint_name resnet18_best.pth
```

Artifacts:
- `models/resnet18_best.pth` – best weights by val accuracy
- `models/classes.json` – index→class mapping used for inference

## Evaluate + Grad‑CAM

Computes accuracy/metrics, saves confusion matrix, and generates Grad‑CAM overlays:

```bash
python -m src.evaluate --data_dir data/val \
	--model_path models/resnet18_best.pth \
	--classes_path models/classes.json \
	--out_dir models/eval --num_cam 6
```

Outputs saved under `models/eval/`:
- `metrics.json` – accuracy and classification report
- `confusion_matrix.png`
- `grad_cam/*.jpg` – overlays with predicted/true labels

## Run the Flask App

Start the web app to upload an image and see the prediction + Grad‑CAM:

```bash
python app/app.py
```

Open http://localhost:5000 in your browser. The app expects `models/resnet18_best.pth` and `models/classes.json` from training.

## Notes

- ResNet‑18 is loaded with ImageNet weights via the modern `ResNet18_Weights.DEFAULT` API.
- Transforms use ImageNet mean/std to match the pretrained backbone.
- Grad‑CAM targets the last block (`model.layer4[-1]`).

