# leakvision/cli/train.py

import json
import logging
import os
from typing import Any, Dict

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from leakvision.config.defaults import DEFAULT_CONFIG, DATA_DIR, DICT_PATH, MANIFEST_PATH
from leakvision.data.manifest_arrays import preparar_dataset
from leakvision.data.splits_manifest_crops import split_from_manifest_with_crops
from leakvision.train.metrics_eval import evaluate
from leakvision.train.pipeline_hybrid_cli import (
    SingleBackboneModel,
    get_transforms,
    set_seed,
    train_one_epoch,
)
from leakvision.viz.run_artifacts import make_run_dir, save_eval_bundle, save_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def remap_8_to_2(y: np.ndarray) -> np.ndarray:
    """
    old 0,1,2,3 -> new 0
    old 4,5,6,7 -> new 1
    """
    y = np.asarray(y, dtype=np.int64)
    mapping = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() >= len(mapping)):
        raise ValueError(f"Labels fuera de rango para mapping: min={y.min()} max={y.max()}")
    return mapping[y]


def build_optimizer_staged(model: nn.Module, config: Dict[str, Any], stage: str) -> torch.optim.Optimizer:
    """
    stage:
      - "head"
      - "head+backbone"

    Asume SingleBackboneModel con:
      - model.head
      - model.backbone
    """
    base_lr = float(config["learning_rate"])
    wd = float(config["weight_decay"])

    param_groups = [{"params": model.head.parameters(), "lr": base_lr}]

    if stage == "head+backbone":
        param_groups.append({"params": model.backbone.parameters(), "lr": base_lr * 0.05})

    return torch.optim.AdamW(param_groups, weight_decay=wd)


def _mlflow_safe_params(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    MLflow params deben ser escalares (str/int/float/bool).
    Convertimos lo demás a str.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main(config: Dict[str, Any]) -> None:
    logger.info(f"Config: {config}")
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Run dir + config enriquecida con rutas (para auditoría)
    config_run = dict(config)
    config_run["data_dir"] = DATA_DIR
    config_run["dict_path"] = DICT_PATH
    config_run["manifest_path"] = MANIFEST_PATH

    run_dir = make_run_dir(config["save_dir"], config_run)
    logger.info(f"RUN DIR: {run_dir}")

    # --- MLflow (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("leakvision_experiments")

    # Run padre
    with mlflow.start_run(run_name=os.path.basename(run_dir)):
        mlflow.log_params(_mlflow_safe_params(config_run))

        # compat (aunque tu pipeline normaliza dentro del engine)
        _ = get_transforms(int(config["image_size"]))

        # --- Data
        data = preparar_dataset(DATA_DIR, DICT_PATH, image_size=int(config["image_size"]))
        images, labels = data.images, data.labels

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            df_split,
            groups_train, groups_val, groups_test,
        ) = split_from_manifest_with_crops(
            images=images,
            labels=labels,
            df_images=data.df,
            manifest_path=MANIFEST_PATH,
            df_image_col="imagen",
            manifest_image_col="file_name",
            split_col="split",
            group_col="anon_slide_id",
            train_name="train",
            val_name="val",
            test_name="test",
        )

        # Remap si num_classes=2
        if int(config["num_classes"]) == 2:
            y_train = remap_8_to_2(y_train)
            y_val = remap_8_to_2(y_val)
            y_test = remap_8_to_2(y_test)

        ds_train = TensorDataset(torch.tensor(X_train).permute(0, 3, 1, 2), torch.tensor(y_train))
        ds_test = TensorDataset(torch.tensor(X_test).permute(0, 3, 1, 2), torch.tensor(y_test))

        n = len(ds_train)
        gkf = GroupKFold(n_splits=int(config["num_folds"]))

        fold_summaries = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(n), y_train, groups_train), 1):
            # Fold run (nested)
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                fold_dir = os.path.join(run_dir, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)

                # anti-leakage por paciente
                train_p = set(groups_train[tr_idx])
                val_p = set(groups_train[va_idx])
                if train_p.intersection(val_p):
                    raise RuntimeError(f"LEAKAGE fold {fold}: pacientes repetidos entre train y val")

                logger.info(f"Fold {fold}: train={len(tr_idx)} val={len(va_idx)} test={len(ds_test)}")

                train_loader = DataLoader(
                    Subset(ds_train, tr_idx),
                    batch_size=int(config["batch_size"]),
                    shuffle=True,
                    num_workers=int(config["num_workers"]),
                )
                val_loader = DataLoader(
                    Subset(ds_train, va_idx),
                    batch_size=int(config["batch_size"]),
                    shuffle=False,
                    num_workers=int(config["num_workers"]),
                )
                test_loader = DataLoader(
                    ds_test,
                    batch_size=int(config["batch_size"]),
                    shuffle=False,
                    num_workers=int(config["num_workers"]),
                )

                # --- Model (SINGLE BACKBONE)
                model = SingleBackboneModel(
                    model_name=str(config["model_name"]),
                    num_classes=int(config["num_classes"]),
                    pretrained=bool(config.get("pretrained", True)),
                    freeze_backbone=True,   # head-only al inicio
                ).to(device)

                logger.info(
                    f"Model: {config['model_name']} | Trainable params start: {count_trainable_params(model):,}"
                )

                # --- Class weights por fold
                y_tr_fold = np.asarray(y_train)[tr_idx]
                C = int(config["num_classes"])
                counts = np.bincount(y_tr_fold, minlength=C).astype(np.float32)
                counts[counts == 0] = 1.0
                weights = counts.sum() / counts
                weights = weights / weights.mean()
                weights = np.clip(weights, 0.25, 4.0)

                criterion = nn.CrossEntropyLoss(
                    weight=torch.tensor(weights, dtype=torch.float32, device=device)
                )

                logger.info(f"Fold {fold} class counts: {counts.tolist()}")
                logger.info(f"Fold {fold} class weights: {weights.tolist()}")

                # --- Optimizer staged
                stage = "head"
                optimizer = build_optimizer_staged(model, config, stage)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(config["step_size"]),
                    gamma=float(config["gamma"]),
                )
                scaler = torch.cuda.amp.GradScaler() if bool(config["mixed_precision"]) else None

                # --- History
                history = {
                    "epoch": [],
                    "train_loss": [],
                    "train_acc": [],
                    "train_balanced_acc": [],
                    "train_f1_macro": [],
                    "val_log_loss": [],
                    "val_acc": [],
                    "val_balanced_acc": [],
                    "val_f1_macro": [],
                }

                best_val_f1 = -1.0
                best_epoch = -1
                best_state = None

                # ---------------------------
                # Early stopping setup (por fold)
                # ---------------------------
                use_es = bool(config.get("early_stopping", False))
                es_metric = str(config.get("early_stopping_metric", "f1_macro"))  # ojo: keys vienen de evaluate()
                es_mode = str(config.get("early_stopping_mode", "max")).lower()
                es_patience = int(config.get("early_stopping_patience", 10))
                es_min_delta = float(config.get("early_stopping_min_delta", 0.0))
                es_min_epochs = int(config.get("early_stopping_min_epochs", 0))
                es_reset_on_unfreeze = bool(config.get("early_stopping_reset_on_unfreeze", True))

                if es_mode not in ("max", "min"):
                    raise ValueError("early_stopping_mode debe ser 'max' o 'min'")

                best_es_value = -float("inf") if es_mode == "max" else float("inf")
                no_improve = 0
                early_stopped = False
                early_stop_epoch = None

                # Epoch para soltar el backbone:
                # prioridad: unfreeze_backbone_epoch, fallback: freeze_epochs, fallback final: 0
                if "unfreeze_backbone_epoch" in config:
                    unfreeze_backbone_epoch = int(config["unfreeze_backbone_epoch"])
                elif "freeze_epochs" in config:
                    unfreeze_backbone_epoch = int(config["freeze_epochs"])
                    logger.warning(
                        "No encuentro 'unfreeze_backbone_epoch' en config; usando freeze_epochs=%d",
                        unfreeze_backbone_epoch,
                    )
                else:
                    unfreeze_backbone_epoch = 0
                    logger.warning(
                        "No encuentro 'unfreeze_backbone_epoch' ni 'freeze_epochs' en config; usando 0"
                    )

                mlflow.log_param("unfreeze_backbone_epoch_used", unfreeze_backbone_epoch)

                for epoch in range(int(config["num_epochs"])):

                    # ---- unfreeze backbone ----
                    if epoch == unfreeze_backbone_epoch:
                        logger.info(f"[Fold {fold}] Unfreeze BACKBONE at epoch {epoch+1}")
                        if hasattr(model, "unfreeze_backbone"):
                            model.unfreeze_backbone()
                        else:
                            for p in model.backbone.parameters():
                                p.requires_grad = True

                        stage = "head+backbone"
                        optimizer = build_optimizer_staged(model, config, stage)
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(config["step_size"]),
                            gamma=float(config["gamma"]),
                        )
                        logger.info(f"Trainable params: {count_trainable_params(model):,}")

                        if use_es and es_reset_on_unfreeze:
                            no_improve = 0

                    # --- train / val
                    train_out = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        scaler,
                        device,
                        bool(config["mixed_precision"]),
                        config=config,
                    )

                    val_out = evaluate(
                        model,
                        val_loader,
                        device,
                        config=config,
                        verbose=False,
                        return_dict=True,
                    )

                    # --- log por época (MLflow)
                    mlflow.log_metric("train_loss", float(train_out["train_loss"]), step=epoch + 1)
                    mlflow.log_metric("train_acc", float(train_out["train_acc"]), step=epoch + 1)
                    mlflow.log_metric("train_f1_macro", float(train_out["train_f1_macro"]), step=epoch + 1)

                    mlflow.log_metric("val_log_loss", float(val_out.get("log_loss", float("nan"))), step=epoch + 1)
                    mlflow.log_metric("val_acc", float(val_out.get("acc", float("nan"))), step=epoch + 1)
                    mlflow.log_metric("val_f1_macro", float(val_out.get("f1_macro", float("nan"))), step=epoch + 1)

                    # --- history
                    history["epoch"].append(epoch + 1)
                    history["train_loss"].append(train_out["train_loss"])
                    history["train_acc"].append(train_out["train_acc"])
                    history["train_balanced_acc"].append(train_out["train_balanced_acc"])
                    history["train_f1_macro"].append(train_out["train_f1_macro"])
                    history["val_log_loss"].append(val_out.get("log_loss"))
                    history["val_acc"].append(val_out.get("acc"))
                    history["val_balanced_acc"].append(val_out.get("balanced_acc"))
                    history["val_f1_macro"].append(val_out.get("f1_macro"))

                    logger.info(
                        f"Fold {fold} | Epoch {epoch+1}: "
                        f"train_acc={train_out['train_acc']:.4f} train_f1={train_out['train_f1_macro']:.4f} | "
                        f"val_acc={val_out.get('acc', float('nan')):.4f} val_f1={val_out.get('f1_macro', float('nan')):.4f}"
                    )

                    # --- best checkpoint by val f1 macro (para cargar luego)
                    val_f1 = val_out.get("f1_macro", float("nan"))
                    if val_f1 == val_f1 and float(val_f1) > best_val_f1:
                        best_val_f1 = float(val_f1)
                        best_epoch = epoch + 1
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                    scheduler.step()

                    # ---------------------------
                    # Early stopping check (AL FINAL, después de log/history/scheduler)
                    # ---------------------------
                    if use_es:
                        current = val_out.get(es_metric, None)
                        if current is None:
                            raise KeyError(
                                f"early_stopping_metric='{es_metric}' no está en val_out. "
                                f"Keys: {list(val_out.keys())}"
                            )

                        current = float(current)
                        if current == current:  # no NaN
                            if es_mode == "max":
                                improved = current > (best_es_value + es_min_delta)
                            else:
                                improved = current < (best_es_value - es_min_delta)

                            if improved:
                                best_es_value = current
                                no_improve = 0
                            else:
                                no_improve += 1

                            mlflow.log_metric("es_no_improve", float(no_improve), step=epoch + 1)
                            mlflow.log_metric(f"es_best_{es_metric}", float(best_es_value), step=epoch + 1)

                            if (epoch + 1) >= es_min_epochs and no_improve >= es_patience:
                                early_stopped = True
                                early_stop_epoch = epoch + 1
                                logger.info(
                                    f"[Fold {fold}] EARLY STOPPING at epoch {early_stop_epoch} "
                                    f"(best {es_metric}={best_es_value:.4f}, patience={es_patience})"
                                )
                                break

                # --- save learning curves
                save_history(history, fold_dir, prefix="fold")

                # --- load best (por val f1 macro)
                if best_state is not None:
                    model.load_state_dict(best_state)
                    logger.info(
                        f"Fold {fold}: loaded BEST checkpoint from epoch {best_epoch} (val_f1={best_val_f1:.4f})"
                    )

                mlflow.log_metric("best_val_f1_macro", float(best_val_f1))
                mlflow.log_metric("best_epoch", float(best_epoch))

                # --- Early stopping metrics (POR FOLD, dentro del run del fold)
                mlflow.log_metric("early_stopped", float(1.0 if early_stopped else 0.0))
                if early_stop_epoch is not None:
                    mlflow.log_metric("early_stop_epoch", float(early_stop_epoch))

                # --- TRAIN bundle
                train_eval_loader = DataLoader(
                    Subset(ds_train, tr_idx),
                    batch_size=int(config["batch_size"]),
                    shuffle=False,
                    num_workers=int(config["num_workers"]),
                )
                train_eval = evaluate(model, train_eval_loader, device, config=config, verbose=False, return_dict=True)
                save_eval_bundle(train_eval, fold_dir, split_name="train", groups=groups_train[tr_idx])

                # --- VAL bundle
                val_best = evaluate(model, val_loader, device, config=config, verbose=False, return_dict=True)
                save_eval_bundle(val_best, fold_dir, split_name="val_best", groups=groups_train[va_idx])

                # --- TEST bundle
                test_out = evaluate(model, test_loader, device, config=config, verbose=False, return_dict=True)
                save_eval_bundle(test_out, fold_dir, split_name="test", groups=groups_test)

                mlflow.log_metric("test_acc", float(test_out["acc"]))
                mlflow.log_metric("test_f1_macro", float(test_out["f1_macro"]))
                mlflow.log_metric("test_auc_ovr_macro", float(test_out.get("auc_ovr_macro", float("nan"))))

                # --- log artifacts del fold
                mlflow.log_artifacts(fold_dir, artifact_path=f"fold_{fold}")

                fold_summary = {
                    "fold": fold,
                    "best_epoch": int(best_epoch),
                    "best_val_f1_macro": float(best_val_f1),
                    "early_stopped": bool(early_stopped),
                    "early_stop_epoch": int(early_stop_epoch) if early_stop_epoch is not None else None,
                    "test_acc": float(test_out["acc"]),
                    "test_f1_macro": float(test_out["f1_macro"]),
                    "test_auc_ovr_macro": float(test_out.get("auc_ovr_macro", float("nan"))),
                }
                fold_summaries.append(fold_summary)

                logger.info(
                    f"[Fold {fold}] TEST acc={fold_summary['test_acc']:.4f} "
                    f"f1={fold_summary['test_f1_macro']:.4f} auc={fold_summary['test_auc_ovr_macro']:.4f}"
                )

        # --- summary global del run (padre)
        summary = {
            "folds": fold_summaries,
            "mean_test_acc": float(np.mean([f["test_acc"] for f in fold_summaries])) if fold_summaries else float("nan"),
            "mean_test_f1_macro": float(np.mean([f["test_f1_macro"] for f in fold_summaries])) if fold_summaries else float("nan"),
            "mean_test_auc_ovr_macro": float(np.nanmean([f["test_auc_ovr_macro"] for f in fold_summaries])) if fold_summaries else float("nan"),
        }

        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary to: {summary_path}")

        # log también el summary + config.json + etc del run_dir
        mlflow.log_artifacts(run_dir, artifact_path="run_files")


if __name__ == "__main__":
    cfg = DEFAULT_CONFIG.copy()
    main(cfg)