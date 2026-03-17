# leakvision/viz/run_artifacts.py

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def make_run_dir(save_root: str, config: Dict[str, Any]) -> str:
    """
    Crea carpeta única por run:
      checkpoints/runs/20260227_153012_ab12cd34/
    """
    os.makedirs(save_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_str = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    h = hashlib.sha1(cfg_str).hexdigest()[:8]
    run_dir = os.path.join(save_root, "runs", f"{stamp}_{h}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=_json_default)

    return run_dir


def save_history(history: Dict[str, Sequence], out_dir: str, prefix: str = "train") -> None:
    """
    history: dict con listas (epoch, train_loss, val_acc, etc.)
    Guarda CSV + PNG (learning curves).
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}_history.csv")
    keys = list(history.keys())

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = max(len(history[k]) for k in keys) if keys else 0
        for i in range(n):
            row = [history[k][i] if i < len(history[k]) else "" for k in keys]
            w.writerow(row)

    # Curvas (loss y f1/acc si existen)
    png_path = os.path.join(out_dir, f"{prefix}_learning_curves.png")
    epochs = history.get("epoch", list(range(1, len(next(iter(history.values()), [])) + 1)))

    plt.figure()
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="train_loss")
    if "val_log_loss" in history:
        plt.plot(epochs, history["val_log_loss"], label="val_log_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # otra figura: acc/f1
    png2 = os.path.join(out_dir, f"{prefix}_acc_f1.png")
    plt.figure()
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], label="train_acc")
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], label="val_acc")
    if "train_f1_macro" in history:
        plt.plot(epochs, history["train_f1_macro"], label="train_f1_macro")
    if "val_f1_macro" in history:
        plt.plot(epochs, history["val_f1_macro"], label="val_f1_macro")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.legend(); plt.tight_layout()
    plt.savefig(png2, dpi=150)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_dir: str, name: str = "cm") -> None:
    os.makedirs(out_dir, exist_ok=True)
    cm = np.asarray(cm, dtype=int)

    np.save(os.path.join(out_dir, f"{name}.npy"), cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title("Confusion Matrix (counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    # números encima
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    plt.close(fig)

def save_per_class_metrics(results: Dict[str, Any], out_dir: str) -> None:
    """
    results viene de evaluate(return_dict=True)
    Guarda CSV con precision/recall/f1/support por clase.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = np.asarray(results.get("per_class_precision", []), dtype=float)
    r = np.asarray(results.get("per_class_recall", []), dtype=float)
    f = np.asarray(results.get("per_class_f1", []), dtype=float)
    s = np.asarray(results.get("per_class_support", []), dtype=float)

    csv_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i in range(len(p)):
            w.writerow([i, p[i], r[i], f[i], s[i]])


def save_predictions(results: Dict[str, Any], out_dir: str, groups: Optional[Sequence[str]] = None) -> None:
    """
    Guarda y_true, y_pred y probs (N,C) para auditoría.
    """
    os.makedirs(out_dir, exist_ok=True)
    ys = np.asarray(results["ys"], dtype=int)
    ps = np.asarray(results["ps"], dtype=int)
    pr = np.asarray(results["pr"], dtype=float)  # (N,C)

    np.save(os.path.join(out_dir, "y_true.npy"), ys)
    np.save(os.path.join(out_dir, "y_pred.npy"), ps)
    np.save(os.path.join(out_dir, "probs.npy"), pr)

    # CSV compacto (ojo: puede pesar si N grande; aquí N~3000 test ok)
    csv_path = os.path.join(out_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["idx", "y_true", "y_pred"] + [f"p{i}" for i in range(pr.shape[1])]
        if groups is not None:
            header = ["idx", "group"] + header[1:]
        w.writerow(header)

        for i in range(len(ys)):
            row = [i, ys[i], ps[i]] + pr[i].tolist()
            if groups is not None:
                row = [i, groups[i]] + row[1:]
            w.writerow(row)


def save_roc_pr_curves(results: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ys = np.asarray(results["ys"], dtype=int)
    pr = np.asarray(results["pr"], dtype=float)  # (N,C)
    C = pr.shape[1]

    # ✅ binarización robusta (también sirve para C=2)
    y_bin = (ys.reshape(-1, 1) == np.arange(C).reshape(1, -1)).astype(np.int32)  # (N,C)

    roc_data = {}
    pr_data = {}

    # ROC por clase
    plt.figure()
    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], pr[:, c])
        roc_auc = auc(fpr, tpr)
        roc_data[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
        plt.plot(fpr, tpr, label=f"class {c} AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC OvR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_ovr.png"), dpi=150)
    plt.close()

    # PR por clase
    plt.figure()
    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, c], pr[:, c])
        ap = average_precision_score(y_bin[:, c], pr[:, c])
        pr_data[c] = {"precision": prec.tolist(), "recall": rec.tolist(), "ap": float(ap)}
        plt.plot(rec, prec, label=f"class {c} AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR OvR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_ovr.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "roc_ovr.json"), "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2, default=_json_default, ensure_ascii=False)
    with open(os.path.join(out_dir, "pr_ovr.json"), "w", encoding="utf-8") as f:
        json.dump(pr_data, f, indent=2, default=_json_default, ensure_ascii=False)


def save_eval_bundle(results: Dict[str, Any], out_dir: str, split_name: str, groups: Optional[Sequence[str]] = None) -> None:
    """
    Paquete completo por split (val/test):
      - metrics.json
      - classification_report.txt
      - confusion matrix (npy+png)
      - per_class_metrics.csv
      - roc/pr (png+json)
      - predictions (npy+csv)
    """
    os.makedirs(out_dir, exist_ok=True)
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # JSON (todo lo scalarizable)
    with open(os.path.join(split_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    # report
    rep = results.get("classification_report", "")
    with open(os.path.join(split_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    # CM
    cm = np.asarray(results.get("confusion_matrix"))
    save_confusion_matrix(cm, split_dir, name="confusion_matrix")

    # per class
    save_per_class_metrics(results, split_dir)

    # ROC/PR
    if "pr" in results and len(results["pr"]) > 0:
        save_roc_pr_curves(results, split_dir)

    # preds
    save_predictions(results, split_dir, groups=groups)