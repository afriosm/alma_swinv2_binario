import re

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# splits_manifest_crops.py
#
# Este módulo implementa un split Train/Test basado en un archivo "manifest"
# (CSV) cuando las imágenes del dataset incluyen variantes tipo crops con sufijos
# como: _crop_0, _crop_12, etc.
#
# Idea central:
#  - El manifest define el split ("train"/"test") para el nombre de imagen base.
#  - Cada imagen crop hereda el split de su imagen base.
#
# Componentes:
#  - _CROP_RE: regex que detecta el sufijo "_crop_<n>" justo antes de la extensión.
#  - base_name(): remueve el sufijo de crop y retorna el nombre base del archivo.
#  - split_from_manifest_with_crops(): cruza df_images (dataset real) con el
#    manifest usando el nombre base y luego separa imágenes/labels según split.
#
# Entradas principales:
#  - images (np.ndarray): arreglo N x H x W x C (o similar) alineado con df_images
#  - labels (np.ndarray): vector (N,) alineado con df_images
#  - df_images (pd.DataFrame): debe contener columna con el nombre del archivo
#  - manifest_path (str): ruta al CSV con columnas file_name y split (por defecto)
#
# Salida:
#  - X_train, X_test, y_train, y_test, df_full (DataFrame merged con el split)
# -----------------------------------------------------------------------------

_CROP_RE = re.compile(r"_crop_\d+(?=\.[^.]+$)", flags=re.IGNORECASE)

def base_name(fname: str) -> str:
    """
    Quita sufijos tipo _crop_0, _crop_12 justo antes de la extensión.
    Ej:
      A03_ASC-US_x12088_y123417_crop_0.jpg -> A03_ASC-US_x12088_y123417.jpg
    """
    fname = str(fname).strip()
    return _CROP_RE.sub("", fname)

def split_from_manifest_with_crops(
    images: np.ndarray,
    labels: np.ndarray,
    df_images: pd.DataFrame,
    manifest_path: str,
    df_image_col: str = "imagen",
    manifest_image_col: str = "file_name",
    split_col: str = "split",
    group_col: str = "anon_slide_id",   # <-- NUEVO
    train_name: str = "train",
    val_name: str = "val",              # <-- NUEVO
    test_name: str = "test",
):
    """
    Split por manifest cuando tu dataset tiene crops (_crop_0, _crop_1, ...).

    Lógica:
    - manifest define split por imagen base (file_name)
    - cada crop hereda el split de su base

    Retorna:
    X_train, X_test, y_train, y_test, df_merged
    """

    df_manifest = pd.read_csv(manifest_path)

    # Validación columnas
    for col in (manifest_image_col, split_col):
        if col not in df_manifest.columns:
            raise KeyError(f"El manifest NO tiene '{col}'. Columnas: {df_manifest.columns.tolist()}")
    if df_image_col not in df_images.columns:
        raise KeyError(f"df_images NO tiene '{df_image_col}'. Columnas: {df_images.columns.tolist()}")

    # Base name en ambos
    keep_cols = [manifest_image_col, split_col]
    if group_col in df_manifest.columns:
        keep_cols.append(group_col)

    keep_cols = [manifest_image_col, split_col]
    if group_col in df_manifest.columns:
        keep_cols.append(group_col)

    df_m = df_manifest[keep_cols].copy()
    df_m["base"] = df_m[manifest_image_col].astype(str).map(base_name)

    df_d = df_images.copy()
    df_d["base"] = df_d[df_image_col].astype(str).map(base_name)

    # Merge por base
    merge_cols = ["base", split_col] + ([group_col] if group_col in df_m.columns else [])
    df_full = df_d.merge(df_m[merge_cols], on="base", how="left")

    # Chequeo: cuántas crops quedaron sin split
    missing = df_full[df_full[split_col].isna()]
    if len(missing) > 0:
        print(f" {len(missing)} crops no encontraron split en el manifest.")
        print("Ejemplos (hasta 10):", missing[df_image_col].head(10).tolist())

    # Filtrar solo las que sí tienen split
    df_full = df_full.dropna(subset=[split_col]).reset_index(drop=False)  # guarda el índice original

    # Índices originales (alineados con images/labels)
    idx_orig = df_full["index"].to_numpy(dtype=int)

    # Separar indices por split
    train_idx = idx_orig[df_full[split_col].to_numpy() == train_name]
    test_idx  = idx_orig[df_full[split_col].to_numpy() == test_name]
    spl = df_full[split_col].to_numpy()
    train_mask = (spl == train_name)
    val_mask   = (spl == val_name)
    test_mask  = (spl == test_name)

    train_idx = idx_orig[train_mask]
    val_idx   = idx_orig[val_mask]
    test_idx  = idx_orig[test_mask]

    X_train, y_train = images[train_idx], labels[train_idx]
    X_val,   y_val   = images[val_idx],   labels[val_idx]
    X_test,  y_test  = images[test_idx],  labels[test_idx]

    groups_train = None
    groups_val = None
    groups_test = None
    if group_col in df_full.columns:
        groups_train = df_full.loc[train_mask, group_col].astype(str).to_numpy()
        groups_val   = df_full.loc[val_mask,   group_col].astype(str).to_numpy()
        groups_test  = df_full.loc[test_mask,  group_col].astype(str).to_numpy()
    if len(train_idx) == 0:
        raise ValueError("No hay filas 'train' después del cruce (ni siquiera por base).")
    if len(test_idx) == 0:
        raise ValueError("No hay filas 'test' después del cruce (ni siquiera por base).")

    X_train, y_train = images[train_idx], labels[train_idx]
    X_test,  y_test  = images[test_idx],  labels[test_idx]

    print("Split por manifest (con crops) OK")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, df_full, groups_train, groups_val, groups_test