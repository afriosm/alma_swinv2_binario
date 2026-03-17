import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_curve

# -----------------------------------------------------------------------------
# metrics_eval.py
#
# Este módulo implementa una evaluación extendida para modelos de clasificación
# (binaria o multiclase) a partir de un DataLoader de PyTorch.
#
# Funcionalidad principal:
#  - _topk_accuracy(): calcula top-k accuracy usando la matriz de probabilidades.
#  - _expected_calibration_error(): calcula ECE (Expected Calibration Error)
#    binning sobre la confianza máxima (max prob por muestra).
#  - evaluate(): corre inferencia en el loader (sin gradientes), aplica
#    preprocesamiento estándar (uint8 -> float [0,1], resize, normalización
#    estilo ImageNet), obtiene probabilidades softmax y calcula un conjunto amplio
#    de métricas:
#      * accuracy, balanced accuracy
#      * F1 macro y weighted
#      * precision/recall macro y weighted
#      * top-3/top-5 accuracy (si aplica)
#      * log loss
#      * AUC OvR macro (si aplica) y Average Precision OvR macro (si aplica)
#      * matriz de confusión
#      * métricas por clase (precision/recall/F1/support)
#      * reporte de clasificación (texto)
#      * métricas de calibración (ECE)
#      * estadísticas de confianza en aciertos vs errores
#      * distribuciones de clases verdaderas y predichas (para diagnosticar colapso)
#
# Notas:
#  - Si config es None, intenta usar una variable global "config" si existe.
#  - Por compatibilidad, puede retornar un dict rico (default) o una tupla
#    (acc, f1_macro, auc, ys, ps, pr) si return_dict=False.
# -----------------------------------------------------------------------------

def _topk_accuracy(y_true, prob_matrix, k=3):
    # y_true: (N,), prob_matrix: (N,C)
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix)
    topk = np.argsort(prob_matrix, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

def _expected_calibration_error(y_true, prob_matrix, n_bins=15):
    """
    ECE (Expected Calibration Error) con bins sobre la confianza max.
    """
    y_true = np.asarray(y_true)
    prob_matrix = np.asarray(prob_matrix)
    conf = prob_matrix.max(axis=1)
    pred = prob_matrix.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)

@torch.no_grad()
def evaluate(model, loader, device, config=None, verbose=True, return_dict=True):
    """
    Evaluación extendida:
    - acc, balanced_acc
    - f1 macro/weighted + precision/recall macro/weighted
    - top-3 / top-5 accuracy (si C>=5)
    - logloss
    - AUC OvR macro (si aplica)
    - AP (average precision) macro OvR (si aplica)
    - confusion matrix
    - per-class metrics
    - calibration (ECE)
    - confidence stats (aciertos vs errores)
    
    Retorna por defecto un dict con todo + ys, ps, pr.
    Si return_dict=False, retorna (acc, f1_macro, auc, ys, ps, pr) para compatibilidad.
    """
    model.eval()
    
    # Si config es None, intenta usar variable global config si existe
    if config is None:
        try:
            config = globals().get("config", {})
        except Exception:
            config = {}

    num_classes = int(config.get("num_classes", 2))
    image_size = int(config.get("image_size", 256))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    ys, ps = [], []
    probs_all = []

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

        logits = model(imgs)
        prob = torch.softmax(logits, dim=1)  # (N,C)
        pred = prob.argmax(dim=1)

        ys.extend(labels.detach().cpu().tolist())
        ps.extend(pred.detach().cpu().tolist())
        probs_all.append(prob.detach().cpu())

    ys = np.asarray(ys, dtype=np.int64)
    ps = np.asarray(ps, dtype=np.int64)
    prob_matrix = torch.cat(probs_all, dim=0).numpy() if len(probs_all) else np.zeros((0, num_classes))

    # --- Métricas globales
    acc = float(accuracy_score(ys, ps))
    bal_acc = float(balanced_accuracy_score(ys, ps))

    f1_macro = float(f1_score(ys, ps, average="macro", zero_division=0))
    f1_weighted = float(f1_score(ys, ps, average="weighted", zero_division=0))

    prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
        ys, ps, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, _, _ = precision_recall_fscore_support(
        ys, ps, average="weighted", zero_division=0
    )

    # Top-k (solo multiclase real)
    top3 = _topk_accuracy(ys, prob_matrix, k=3) if num_classes >= 3 and len(ys) else np.nan
    top5 = _topk_accuracy(ys, prob_matrix, k=5) if num_classes >= 5 and len(ys) else np.nan

    # Log loss (cross-entropy en términos de probas)
    # Ojo: log_loss requiere labels presentes o especificar labels=
    labels_space = list(range(num_classes))
    try:
        ll = float(log_loss(ys, prob_matrix, labels=labels_space))
    except Exception:
        ll = np.nan

    # Confusion matrix
    cm = confusion_matrix(ys, ps, labels=labels_space)

    # Per-class metrics
    per_class = precision_recall_fscore_support(ys, ps, labels=labels_space, zero_division=0)
    per_prec, per_rec, per_f1, per_sup = [np.asarray(x) for x in per_class]

    # Calibration / confidence
    if len(ys):
        conf = prob_matrix.max(axis=1)
        correct = (ps == ys)
        conf_correct = float(conf[correct].mean()) if correct.any() else np.nan
        conf_wrong = float(conf[~correct].mean()) if (~correct).any() else np.nan
        ece = _expected_calibration_error(ys, prob_matrix, n_bins=15)
    else:
        conf_correct = conf_wrong = ece = np.nan

    # --- AUC y AP (solo si hay >=2 clases presentes en y_true)
    present = sorted(set(ys.tolist()))
    auc_ovr = np.nan
    ap_ovr = np.nan
    if len(present) >= 2 and len(ys):
        try:
            # y_true binarizado sobre clases presentes
            y_true_bin = label_binarize(ys, classes=present)
            y_score = prob_matrix[:, present]

            # AUC OvR macro
            auc_ovr = float(roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro"))

            # Average precision macro (OvR)
            # average_precision_score espera y_true_bin (N,K) y scores (N,K)
            ap_ovr = float(average_precision_score(y_true_bin, y_score, average="macro"))
        except Exception:
            auc_ovr = np.nan
            ap_ovr = np.nan

    # Reporte de clasificación (string)
    report = classification_report(
        ys, ps, labels=labels_space, digits=4, zero_division=0
    )

    # Diagnóstico colapso (predice pocas clases)
    pred_unique, pred_counts = np.unique(ps, return_counts=True)
    true_unique, true_counts = np.unique(ys, return_counts=True)
    pred_dist = dict(zip(pred_unique.tolist(), pred_counts.tolist()))
    true_dist = dict(zip(true_unique.tolist(), true_counts.tolist()))

    results = {
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "top3_acc": float(top3) if top3 == top3 else np.nan,
        "top5_acc": float(top5) if top5 == top5 else np.nan,
        "log_loss": ll,
        "auc_ovr_macro": auc_ovr,
        "ap_ovr_macro": ap_ovr,
        "ece": ece,
        "conf_mean_correct": conf_correct,
        "conf_mean_wrong": conf_wrong,
        "confusion_matrix": cm,
        "per_class_precision": per_prec,
        "per_class_recall": per_rec,
        "per_class_f1": per_f1,
        "per_class_support": per_sup,
        "classification_report": report,
        "pred_distribution": pred_dist,
        "true_distribution": true_dist,
        # para compatibilidad con tu pipeline:
        "ys": ys,
        "ps": ps,
        "pr": prob_matrix,  # (N,C)
    }

    if verbose:
        print(report)
        print("Acc:", acc, "| BalAcc:", bal_acc, "| F1(macro):", f1_macro, "| F1(weighted):", f1_weighted)
        print("Top3:", top3, "| Top5:", top5, "| LogLoss:", ll)
        print("AUC OvR macro:", auc_ovr, "| AP OvR macro:", ap_ovr)
        print("ECE:", ece, "| Conf(correct):", conf_correct, "| Conf(wrong):", conf_wrong)
        print("True dist:", true_dist)
        print("Pred dist:", pred_dist)

    # compatibilidad con tu uso anterior: acc, f1, auc, ys, ps, pr
    if not return_dict:
        return acc, f1_macro, auc_ovr, ys.tolist(), ps.tolist(), prob_matrix.tolist()

    return results

 # ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_metrics(ys, ps, pr, title, config=None):
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.show()

    # ROC solo si es binario (pr es lista de floats)
    if len(pr) > 0 and not isinstance(pr[0], (list, tuple, np.ndarray)):
        fpr, tpr, _ = roc_curve(ys, pr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(ys, pr):.3f}')
        plt.plot([0, 1], [0, 1], '--')
        plt.title(f'{title} ROC')
        plt.legend()
        plt.show()
    else:
        num_classes = int(config.get("num_classes", 2))
        y_true_bin = label_binarize(ys, classes=list(range(num_classes)))
        y_score = np.asarray(pr)
        auc = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")
        print(f"{title} Macro AUC (OvR): {auc:.4f}")
