# Intelligent Image Classification

A practical image classification pipeline using transfer learning (ResNet-18), with evaluation metrics and Grad-CAM visualizations for model transparency.

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


