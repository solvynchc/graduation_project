from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = ROOT_DIR.parent / "database"

# default paths under ../database
DATA_ROOT = DATABASE_DIR / "YOLO_Dataset_Ready"
PREDICT_SOURCE = DATABASE_DIR / "predict_input"
RAW_MASK_DIR = DATABASE_DIR / "ISIC2018_Task1_Training_GroundTruth"
RAW_MASK_ZIP = DATABASE_DIR / "ISIC2018_Task1_Training_GroundTruth.zip"

# default training and validation settings
PRETRAINED = "yolo11n-seg.pt"
MODEL_YAML = Path("source/yolo11n_seg_dysample_g8_scope.yaml")
MODEL_YAML_RAW_AUX = Path("source/yolo11n_seg_dysample_g8_scope_raw_aux.yaml")
MODEL_YAML_RAW_AUX_EDGE = Path("source/yolo11n_seg_dysample_g8_scope_raw_aux_edge.yaml")
MODEL_YAML_RAW_AUX_EDGE_P2 = Path("source/yolo11n_seg_dysample_g8_scope_raw_aux_edge_p2.yaml")
FINAL_MODEL = Path("runs/yolo11n_seg_dysample_g8_scope/weights/best.pt")

EPOCHS = 50
IMGSZ = 640
BATCH = 8
DEVICE = "0"
WORKERS = 2
OPTIMIZER = "AdamW"
LR0 = 0.001
LRF = 0.01
WEIGHT_DECAY = 0.0005
PATIENCE = 15
SEED = 42

TRAIN_PROJECT = "runs"
TRAIN_NAME = "yolo11n_seg_dysample_g8_scope_pycharm"
TRAIN_NAME_RAW_AUX = "yolo11n_seg_dysample_g8_scope_raw_aux"
TRAIN_NAME_RAW_AUX_EDGE = "yolo11n_seg_dysample_g8_scope_raw_aux_edge"
TRAIN_NAME_RAW_AUX_EDGE_P2 = "yolo11n_seg_dysample_g8_scope_raw_aux_edge_p2"
PREDICT_PROJECT = "predict_results"
PREDICT_NAME = "g8_scope_predict"
CONF = 0.25
