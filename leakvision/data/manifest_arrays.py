import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------------------------------------------------------
# manifest_arrays.py
#
# Este módulo se encarga de construir un dataset “en memoria” a partir de:
#  - Una carpeta con imágenes (jpg/jpeg/png)
#  - Un diccionario en Excel que mapea la columna "Etiqueta" -> "ID_label"
#
# Funcionalidades principales:
#  1) Extraer la etiqueta desde el nombre del archivo (convención con "_").
#  2) Cargar el diccionario de etiquetas a IDs desde un Excel.
#  3) Leer y redimensionar imágenes a un tamaño fijo y convertirlas a RGB.
#  4) Construir un DataFrame con metadatos (imagen, etiqueta, ID, path).
#  5) Retornar un contenedor (DatasetArrays) con:
#      - images: np.ndarray de forma (N, image_size, image_size, 3) en uint8
#      - labels: np.ndarray (N,) en int64
#      - df:     pd.DataFrame con el detalle de cada imagen
#
# Nota de diseño:
#  - El bloque final crea variables globales y métricas rápidas (distribución de
#    clases, verificación de rango, conteos). En un proyecto modular, esto suele
#    moverse a un script/CLI o notebook de ejecución, pero aquí se mantiene tal
#    cual para respetar el código original.
# -----------------------------------------------------------------------------

def extraer_etiqueta(nombre_archivo: str) -> Optional[str]:
    base = os.path.splitext(nombre_archivo)[0]
    partes = base.split("_")
    return partes[1].strip() if len(partes) >= 2 else None


def cargar_mapa_etiqueta_a_id(dic_path: str) -> dict[str, int]:
    df_dic = pd.read_excel(dic_path)
    if "Etiqueta" not in df_dic.columns or "ID_label" not in df_dic.columns:
        raise ValueError("El diccionario debe tener columnas 'Etiqueta' e 'ID_label'.")
    df_dic["Etiqueta"] = df_dic["Etiqueta"].astype(str).str.strip()
    df_dic["ID_label"] = pd.to_numeric(df_dic["ID_label"], errors="raise").astype(int)
    return dict(zip(df_dic["Etiqueta"], df_dic["ID_label"]))


def _leer_y_redimensionar(path_img: str, image_size: int) -> np.ndarray:
    img = Image.open(path_img).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Imagen no tiene 3 canales RGB: {path_img} -> shape {arr.shape}")
    return arr


@dataclass
class DatasetArrays:
    images: np.ndarray
    labels: np.ndarray
    df: pd.DataFrame


def preparar_dataset(
    carpeta_imagenes: str,
    dic_path: str,
    image_size: int,
    extensiones: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    drop_missing: bool = True,
) -> DatasetArrays:
    mapa_id = cargar_mapa_etiqueta_a_id(dic_path)

    archivos = [
        f for f in os.listdir(carpeta_imagenes)
        if f.lower().endswith(tuple(e.lower() for e in extensiones))
    ]
    archivos.sort()

    rows = []
    for f in archivos:
        etiqueta = extraer_etiqueta(f)
        id_label = mapa_id.get(etiqueta)
        rows.append({
            "imagen": f,
            "Etiqueta": etiqueta,
            "ID_label": id_label,
            "path": os.path.join(carpeta_imagenes, f),
        })

    df = pd.DataFrame(rows)

    if drop_missing:
        df = df.dropna(subset=["ID_label"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No quedó ninguna imagen válida tras cruzar con el diccionario.")

    if df["ID_label"].isna().any():
        raise ValueError("Hay imágenes sin ID_label. Revisa el diccionario o nombres de archivo.")

    df["ID_label"] = df["ID_label"].astype(int)

    imgs = [_leer_y_redimensionar(p, image_size) for p in df["path"].tolist()]
    images = np.stack(imgs, axis=0).astype(np.uint8)
    labels = df["ID_label"].to_numpy(dtype=np.int64)

    return DatasetArrays(images=images, labels=labels, df=df)


