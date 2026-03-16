# Runs Comparison Summary

| Model | Best Epoch | Precision(M) | Recall(M) | mAP50(M) | mAP50-95(M) | Fitness | Best Weight (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_yolo11n_seg | 34 | 0.9622 | 0.9229 | 0.9752 | 0.7153 | 1.4548 | 5.727 |
| yolo11n_seg_cbam | 34 | 0.9509 | 0.946 | 0.9793 | 0.7052 | 1.4387 | 5.9 |
| yolo11n_seg_cbam_dysample | 34 | 0.9482 | 0.9514 | 0.9759 | 0.7055 | 1.444 | 5.925 |
| yolo11n_seg_dysample | 34 | 0.9528 | 0.9334 | 0.9727 | 0.7134 | 1.459 | 5.752 |
| yolo11n_seg_dysample_g8 | 34 | 0.9678 | 0.926 | 0.9852 | 0.7272 | 1.4801 | 5.776 |
| yolo11n_seg_dysample_g8_scope | 34 | 0.9513 | 0.9417 | 0.978 | 0.7279 | 1.4822 | 5.824 |
| yolo11n_seg_dysample_g8_scope_cbam_backbone | 34 | 0.9354 | 0.9482 | 0.9788 | 0.7098 | 1.457 | 8.912 |
| yolo11n_seg_dysample_g8_scope_cbam_backbone_fix | 34 | 0.9675 | 0.9249 | 0.976 | 0.7164 | 1.4611 | 5.952 |
