import torch

# Se asume que HybridModel viene de:
# from leakvision.train.pipeline_hybrid_cli import HybridModel
# o desde el módulo donde lo hayas definido.
#
# También se asume que:
#  - config ya está cargado
#  - logger ya está configurado

# -----------------------------------------------------------------------------
# model_setup.py
#
# Este módulo se encarga de:
#  1) Detectar automáticamente el dispositivo disponible (CPU o GPU).
#  2) Instanciar el modelo híbrido (ConvNeXt + SwinV2) usando la configuración.
#  3) Activar DataParallel si hay múltiples GPUs disponibles.
#  4) Mover el modelo al dispositivo correspondiente.
#
# Responsabilidad:
#  - Centralizar la inicialización del modelo y la lógica de hardware.
#  - Mantener separado el setup del modelo respecto al entrenamiento y métricas.
#
# Nota:
#  - Depende de:
#       * HybridModel
#       * config
#       * logger
#    definidos previamente en el pipeline principal.
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridModel(
    cnn_name=config["cnn_arch"],
    swin_name=config["swin_arch"],
    num_classes=config["num_classes"],
    freeze_backbones=config.get("freeze_backbones", False),
)

if torch.cuda.device_count() > 1:
    logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)