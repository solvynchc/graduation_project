# Kaggle Experiment Record: g8_scope + raw-mask aux

## Run information

- Date: 2026-03-16
- Platform: Kaggle Notebook
- GPU: Tesla P100 16GB
- Python: 3.12.12
- PyTorch: 2.9.0+cu126
- Ultralytics: 8.4.21
- Experiment name: `yolo11n_seg_dysample_g8_scope_raw_aux_kaggle`
- Command:

```bash
python train_g8_scope_raw_aux.py \
  --data-root /kaggle/input/datasets/chcchc/yolo-isic-2018/YOLO_Dataset_Ready \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 2 \
  --name yolo11n_seg_dysample_g8_scope_raw_aux_kaggle \
  --raw-mask-dir /kaggle/input/datasets/chcchc/isic2018-task1-training-groundtruth/ISIC2018_Task1_Training_GroundTruth
```

## Training summary

- EarlyStopping patience: `15`
- Best epoch: `32`
- Stop epoch: `47`
- Total training time: `2.300 hours`
- Best weight path:
  `/kaggle/working/graduation_project/ultralytics/runs/yolo11n_seg_dysample_g8_scope_raw_aux_kaggle/weights/best.pt`
- Best weight size: `6.2 MB`
- Last weight size: `6.2 MB`

## Validation summary

- Model summary: `YOLO11n_seg_dysample_g8_scope_raw_aux`
- Layers: `118`
- Parameters: `2,921,036`
- GFLOPs: `9.6`

| Metric | Value |
|---|---:|
| Box Precision | 0.955 |
| Box Recall | 0.923 |
| Box mAP50 | 0.970 |
| Box mAP50-95 | 0.728 |
| Mask Precision | 0.964 |
| Mask Recall | 0.906 |
| Mask mAP50 | 0.963 |
| Mask mAP50-95 | 0.714 |

## Inference speed

| Stage | Time per image |
|---|---:|
| Preprocess | 0.6 ms |
| Inference | 2.7 ms |
| Loss | 0.0 ms |
| Postprocess | 2.1 ms |

## Current interpretation

- This run verifies that the `g8_scope + raw-mask aux` pipeline is trainable end-to-end on Kaggle.
- However, the current `Mask mAP50-95 = 0.714` does not exceed the existing `g8_scope` strong baseline in the local ablation table.
- Therefore, `raw-mask aux` alone should be treated as an intermediate experiment rather than the final best model.
- The next recommended step is to continue toward `raw-mask aux + edge head` and then `raw-mask aux + edge head + P2 fusion`.
