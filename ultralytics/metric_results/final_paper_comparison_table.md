# Final Paper Comparison Table

This table is intended for the paper-level cross-model comparison between the final YOLO model and the U-Net baseline.

Comparison rule:

- `Precision` / `Recall` / `IoU` / `Dice` use the unified pixel-level segmentation evaluation.
- `Inference Time` and `FPS` are measured on the same local device.
- `mAP50(M)` and `mAP50-95(M)` are reported for the YOLO model using the re-validated `best.pt` result under the current local environment.
- `U-Net` does not report `mAP` because the baseline is evaluated as a pure segmentation model.

| Model | Params | Model Size (MB) | Precision | Recall | IoU | Dice | mAP50(M) | mAP50-95(M) | Inference Time (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n-seg + DySample g8_scope | 2,884,043 | 5.824 | 0.9031 | 0.9066 | 0.8257 | 0.8924 | 0.9806 | 0.7430 | 86.7126 | 11.5323 |
| U-Net | 7,763,041 | 29.6677 | 0.8848 | 0.7874 | 0.7142 | 0.8333 | N/A | N/A | 101.5396 | 9.8484 |

## Direct conclusions

- The final YOLO model achieves better `Precision`, `Recall`, `IoU`, and `Dice` than the U-Net baseline under the current unified evaluation.
- The final YOLO model is much lighter, with fewer parameters and a much smaller weight file than U-Net.
- The final YOLO model is also faster in local inference, with lower average latency and higher FPS.

## Source notes

- YOLO pixel-level metrics: `metric_results/val_seg_metrics_summary.json`
- YOLO speed metrics: `metric_results/inference_benchmark.json`
- YOLO mAP metrics: current `best.pt` re-validation output (`runs/segment/val7`)
- U-Net metrics: `runs/unet_baseline/summary.json`
