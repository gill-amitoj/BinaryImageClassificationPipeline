#!/usr/bin/env python3
# ----------------------------------------------------------
# Intelligent Image Classification System - Evaluation & Grad-CAM
# Author: Amitoj Singh (CCID: amitoj3)
# ----------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import build_model
from .grad_cam import GradCAM, overlay_heatmap_on_image


def get_val_loader(data_dir: str, batch_size: int = 32, img_size: int = 224) -> Tuple[DataLoader, List[str]]:
	t = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	ds = datasets.ImageFolder(data_dir, transform=t)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
	return loader, ds.classes


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
	model.eval()
	y_true, y_pred = [], []
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			logits = model(x)
			preds = logits.argmax(dim=1).cpu().tolist()
			y_pred.extend(preds)
			y_true.extend(y.cpu().tolist())

	acc = accuracy_score(y_true, y_pred)
	cm = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred, output_dict=True)
	return acc, cm, report


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path):
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
	ax.set_xticklabels(class_names, rotation=45, ha="right")
	ax.set_yticklabels(class_names)
	ax.set_ylabel("True label")
	ax.set_xlabel("Predicted label")

	thresh = cm.max() / 2.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				format(cm[i, j], "d"),
				ha="center",
				va="center",
				color="white" if cm[i, j] > thresh else "black",
			)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, bbox_inches="tight")
	plt.close(fig)


def grad_cam_samples(
	model: nn.Module,
	data_dir: str,
	class_names: List[str],
	device: torch.device,
	out_dir: Path,
	img_size: int = 224,
	num_images: int = 4,
):
	# Dataset that preserves filepaths for reading original RGB image
	t = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	ds = datasets.ImageFolder(data_dir, transform=t)

	target_layer = getattr(model, "layer4")[-1]
	cam = GradCAM(model, target_layer)

	out_dir = out_dir / "grad_cam"
	out_dir.mkdir(parents=True, exist_ok=True)

	# Select evenly spaced samples across dataset
	indices = np.linspace(0, len(ds) - 1, num=min(num_images, len(ds)), dtype=int)
	for idx in indices:
		x, y = ds[idx]
		# Recover original RGB (load from path and resize to same size for overlay)
		path, _ = ds.samples[idx]
		import PIL.Image

		rgb = np.array(PIL.Image.open(path).convert("RGB").resize((img_size, img_size)))

		x = x.unsqueeze(0).to(device)
		with torch.no_grad():
			logits = model(x)
			pred_idx = int(logits.argmax(dim=1).item())
		heatmap = cam.generate(x, class_idx=pred_idx)
		overlay = overlay_heatmap_on_image(rgb, heatmap, alpha=0.45)

		fname = out_dir / f"cam_{idx}_true-{class_names[y]}_pred-{class_names[pred_idx]}.jpg"
		import cv2

		cv2.imwrite(str(fname), overlay)

	cam.remove_hooks()


def main():
	parser = argparse.ArgumentParser(description="Evaluate model and generate Grad-CAM samples")
	parser.add_argument("--data_dir", type=str, default="data/val", help="Directory with validation/test images")
	parser.add_argument("--model_path", type=str, default="models/resnet18_best.pth")
	parser.add_argument("--classes_path", type=str, default="models/classes.json", help="idx->class mapping JSON")
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--out_dir", type=str, default="models/eval")
	parser.add_argument("--num_cam", type=int, default=6, help="How many Grad-CAM samples to save")

	args = parser.parse_args()
	device = torch.device(args.device)

	# Data
	loader, class_names = get_val_loader(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)

	# Classes mapping if exists (ensures order consistency)
	classes_json = Path(args.classes_path)
	if classes_json.exists():
		with open(classes_json, "r", encoding="utf-8") as f:
			idx_to_class = json.load(f)
		# Ensure model head matches expected num classes
		class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

	num_classes = len(class_names)

	# Model
	model = build_model(num_classes)
	model.load_state_dict(torch.load(args.model_path, map_location=device))
	model.to(device)

	# Eval
	acc, cm, report = evaluate_model(model, loader, device)
	print(f"Accuracy: {acc:.4f}")
	print("Classification Report:")
	from pprint import pprint

	pprint(report)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Save metrics
	with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
		json.dump({"accuracy": acc, "report": report}, f, indent=2)
	save_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

	# Grad-CAM examples
	grad_cam_samples(model, args.data_dir, class_names, device, out_dir, img_size=args.img_size, num_images=args.num_cam)

	print(f"Saved evaluation artifacts in {out_dir}")


if __name__ == "__main__":
	main()

