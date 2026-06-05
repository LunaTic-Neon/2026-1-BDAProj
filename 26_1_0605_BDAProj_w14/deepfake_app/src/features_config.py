# src/features_config.py
# Face detection / feature extraction configuration. 변경하여 튜닝 가능합니다.

# Use GPU if available and torch is installed. Set to True to prefer GPU.
USE_GPU = False

# Minimum face size (pixels) for detector to consider. Increase to ignore tiny faces.
MIN_FACE_SIZE = 40

# MTCNN thresholds (if applicable) - a tuple of three floats for [P-Net, R-Net, O-Net]
MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)

# RetinaFace score threshold (if used)
RETINAFACE_THRESHOLD = 0.6

# Device selection helper
try:
    import torch
    DEVICE = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'
