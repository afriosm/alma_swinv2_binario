# leakvision/config/defaults.py

DEFAULT_CONFIG: dict = {
    "seed": 42,
    "image_size": 256,
    "batch_size": 16,
    "num_workers": 4,

    # ---- modelo simple (1 backbone)
    "model_name": "swinv2_base_window8_256",     # <- backbone único (timm)
    "pretrained": True,
    "num_classes": 2,

    # ---- entrenamiento
    "num_epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "step_size": 15,
    "gamma": 0.9,
    "mixed_precision": True,
    "unfreeze_backbone_epoch": 15,
    # ---- early stopping
    "early_stopping": False,
    "early_stopping_metric": "log_loss",  # o "val_log_loss"
    "early_stopping_mode": "min",             # "max" si f1/acc, "min" si loss
    "early_stopping_patience": 8,             # epochs sin mejora
    "early_stopping_min_delta": 1e-3,         # mejora mínima para contar como mejora real
    "early_stopping_min_epochs": 10,          # no parar antes de esto
    "early_stopping_reset_on_unfreeze": True, # resetea paciencia cuando descongelas
    # ---- CV / split
    "num_folds": 2,
    "test_size": 0.3,

    # ---- outputs
    "save_dir": "./checkpoints",
}

config = DEFAULT_CONFIG.copy()

# ---------------- RUTAS ----------------
DATA_DIR = "/home/kmtorres/Documentos/Documents/PROYECTOS/PROYECTOS_ALMA/ALMA/swin_transformer_conv_next/crops_256"
DICT_PATH = "/home/kmtorres/Documentos/Documents/PROYECTOS/PROYECTOS_ALMA/ALMA/DATA/dataset_POLICARPA_EPITELIALES/dictionary/diccionario.xlsx"
MANIFEST_PATH = "/home/kmtorres/Documentos/Documents/PROYECTOS/PROYECTOS_ALMA/ALMA/DATA/dataset_POLICARPA_EPITELIALES_split/split_manifest.csv"