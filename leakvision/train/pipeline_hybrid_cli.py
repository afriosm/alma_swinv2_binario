import argparse
import json
import logging
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms

from leakvision.config.defaults import DEFAULT_CONFIG
from leakvision.config.defaults import DATA_DIR, DICT_PATH, MANIFEST_PATH


import timm
from captum.attr import IntegratedGradients, LayerGradCam

# -----------------------------------------------------------------------------
# pipeline_hybrid_cli.py
#
# Este módulo contiene el “core” del pipeline de un modelo híbrido ConvNeXt + SwinV2
# e integra (en un solo lugar) varias piezas:
#
# 1) Lectura de configuración por CLI:
#    - Usa argparse para recibir una ruta a un archivo de configuración (JSON/YAML).
#    - Parte de un DEFAULT_CONFIG y lo actualiza con el contenido del archivo del usuario.
#    - Crea el directorio de guardado y configura logging.
#
# 2) Utilidades:
#    - set_seed(): fija semillas para reproducibilidad en Python, NumPy y PyTorch.
#    - get_transforms(): define transformaciones de entrenamiento y prueba usando torchvision.
#
# 3) Modelo:
#    - HybridModel: crea dos backbones con timm (ConvNeXt y SwinV2), concatena sus
#      embeddings y aplica un head lineal para clasificar.
#    - Opción de congelar backbones (freeze_backbones).
#
# 4) Entrenamiento y evaluación:
#    - train_one_epoch(): loop de entrenamiento con soporte de AMP, normalización
#      estilo ImageNet y resize a config['image_size'] usando interpolate.
#    - evaluate(): loop de evaluación, cálculo de accuracy, F1 y AUC para binario
#      o multiclase, retornando también y_true, y_pred y probabilidades.
#
# 5) Explainability:
#    - explain_image(): calcula Integrated Gradients y Grad-CAM (LayerGradCam)
#      sobre una muestra, aplicando el mismo preprocesamiento que en entrenamiento.
#
# Nota de modularización “pro”:
#  - En una estructura más limpia, esta celda se separa en módulos:
#    * config.py (carga de config + logging)
#    * transforms.py (get_transforms)
#    * hybrid.py (HybridModel)
#    * engine.py (train_one_epoch/evaluate)
#    * explain.py (explain_image)
#  - Aquí se mantiene todo junto y el código no se altera para respetar el original.
# -----------------------------------------------------------------------------

# CLI and config
parser = argparse.ArgumentParser(description="Hybrid ConvNeXt + SwinV2 Pipeline")
parser.add_argument("--config", type=str, help="Path to JSON/YAML config file")
# args = parser.parse_args()
args, _ = parser.parse_known_args()
config = DEFAULT_CONFIG.copy()
if args.config:
    ext = os.path.splitext(args.config)[1].lower()
    with open(args.config) as f:
        user_cfg = yaml.safe_load(f) if ext in (".yaml", ".yml") else json.load(f)
    config.update(user_cfg)

os.makedirs(config['save_dir'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(size: int) -> Dict[str, transforms.Compose]:
    """Return training and testing torchvision transforms."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


# ----------------------------------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------------------------------
# leakvision/train/pipeline_hybrid_cli.py

class SingleBackboneModel(nn.Module):
    """
    Modelo 1-backbone para auditoría limpia.
    Mantiene interfaz similar:
      - self.backbone
      - self.head
      - freeze_backbone()
      - unfreeze_backbone()
    """
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, num_classes),
        )
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

@torch.no_grad()
def _batch_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    """
    Métricas rápidas por batch (sin sklearn):
      - accuracy
      - balanced accuracy (macro recall)
      - f1 macro (aprox por batch)
    Nota: f1 macro por batch es una aproximación; para métricas “oficiales” usa evaluate().
    """
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    acc = correct.mean().item()

    # balanced acc = mean recall por clase presente en el batch
    num_classes = logits.shape[1]
    recalls = []
    f1s = []

    for c in range(num_classes):
        mask_true = (labels == c)
        if mask_true.any():
            tp = ((preds == c) & mask_true).sum().item()
            fn = ((preds != c) & mask_true).sum().item()
            recall_c = tp / (tp + fn + 1e-12)
            recalls.append(recall_c)

            # precision para f1
            mask_pred = (preds == c)
            fp = (mask_pred & (~mask_true)).sum().item()
            prec_c = tp / (tp + fp + 1e-12)
            f1_c = 2 * prec_c * recall_c / (prec_c + recall_c + 1e-12)
            f1s.append(f1_c)

    bal_acc = float(np.mean(recalls)) if len(recalls) else np.nan
    f1_macro = float(np.mean(f1s)) if len(f1s) else np.nan
    return acc, bal_acc, f1_macro


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, config=None):
    """
    Entrena 1 epoch y retorna un dict con:
      - train_loss
      - train_acc
      - train_balanced_acc
      - train_f1_macro

    Cambios vs versión anterior:
      - Ya no depende de `config` global: recibe `config` (o intenta buscarlo).
      - Calcula métricas rápidas en train para curvas de aprendizaje.
      - Soporta AMP tanto con scaler como sin scaler.
    """
    model.train()

    # Si config es None, intenta usar variable global config si existe
    if config is None:
        try:
            config = globals().get("config", {})
        except Exception:
            config = {}

    image_size = int(config.get("image_size", 256))

    total_loss = 0.0
    total_acc = 0.0
    total_bal_acc = 0.0
    total_f1_macro = 0.0
    n_batches = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Asegurar float en [0,1]
        if imgs.dtype == torch.uint8:
            imgs = imgs.float().div_(255.0)
        else:
            imgs = imgs.float()

        # Resize si hace falta
        if imgs.shape[-1] != image_size or imgs.shape[-2] != image_size:
            imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bilinear", align_corners=False)

        # Normalización estándar
        imgs = (imgs - mean) / std

        optimizer.zero_grad(set_to_none=True)

        if amp:
            # AMP moderno recomendado es torch.amp.autocast('cuda'), pero dejamos compatibilidad.
            with torch.cuda.amp.autocast(True):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

        # Métricas de entrenamiento (aprox)
        with torch.no_grad():
            acc_b, bal_b, f1_b = _batch_metrics_from_logits(logits.detach(), labels)
            total_acc += acc_b
            total_bal_acc += (bal_b if bal_b == bal_b else 0.0)   # evita nan
            total_f1_macro += (f1_b if f1_b == f1_b else 0.0)

        n_batches += 1

    if n_batches == 0:
        return {
            "train_loss": np.nan,
            "train_acc": np.nan,
            "train_balanced_acc": np.nan,
            "train_f1_macro": np.nan,
        }

    return {
        "train_loss": total_loss / n_batches,
        "train_acc": total_acc / n_batches,
        "train_balanced_acc": total_bal_acc / n_batches,
        "train_f1_macro": total_f1_macro / n_batches,
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, pr = [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    num_classes = int(config.get("num_classes", 2))

    for imgs, labels in loader:
        target = config["image_size"]

        imgs = imgs.to(device, non_blocking=True).float().div_(255.0)

        if imgs.shape[-1] != target or imgs.shape[-2] != target:
            imgs = F.interpolate(imgs, size=(target, target), mode="bilinear", align_corners=False)

        imgs = (imgs - mean) / std
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)          # (N, C)
        preds = probs.argmax(1)                       # (N,)

        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())

        if num_classes == 2:
            pr.extend(probs[:, 1].cpu().tolist())     # lista de floats
        else:
            pr.extend(probs.cpu().tolist())           # lista de listas (N x C)

    acc = accuracy_score(ys, ps)

    if num_classes == 2:
        f1  = f1_score(ys, ps, average="binary")
        auc = roc_auc_score(ys, pr)
    else:
        # F1 para multiclase: macro (trata todas las clases “igual”)
        f1 = f1_score(ys, ps, average="macro")

        # AUC multiclase (One-vs-Rest) requiere scores (N,C) y y_true binarizado
        y_true_bin = label_binarize(ys, classes=list(range(num_classes)))
        y_score = np.asarray(pr)  # (N, C)
        auc = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    return acc, f1, auc, ys, ps, pr
# Explainability
# ----------------------------------------------------------------------------
def explain_image(model, img, label, device) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients and Grad-CAM for one sample."""
    ig = IntegratedGradients(model)
    lg = LayerGradCam(model, model.swin.layers[-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    batch = img.unsqueeze(0).to(device)

    # 1) si viene uint8, pásalo a float y escálalo a [0,1]
    if batch.dtype == torch.uint8:
        batch = batch.float().div_(255.0)
    else:
        batch = batch.float()

    # 2) redimensiona al image_size del modelo si aplica
    target = config["image_size"]
    if batch.shape[-1] != target or batch.shape[-2] != target:
        batch = F.interpolate(batch, size=(target, target), mode="bilinear", align_corners=False)

    # 3) normaliza igual que en entrenamiento
    batch = (batch - mean) / std
    ig_attr, _ = ig.attribute(batch, target=label, return_convergence_delta=True)
    gc_attr = lg.attribute(batch, target=label)
    return (
    ig_attr.squeeze(0).detach().cpu().numpy(),
    gc_attr.squeeze(0).detach().cpu().numpy()
)