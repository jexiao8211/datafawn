# Pose Estimation

The pose estimation stage tracks body parts in videos using DeepLabCut models. This is typically the first step in the pipeline when starting from raw video footage.

## Classes

### `DeepLabCutPoseEstimator`

A wrapper around DeepLabCut's SuperAnimal inference models. It handles the loading of models and running inference on video files.

**Location:** `datafawn.pose_estimators`

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `'superanimal_quadruped'` | The name of the SuperAnimal model to use (e.g., 'superanimal_quadruped'). |
| `detector_name` | `str` | `'fasterrcnn_resnet50_fpn_v2'` | The object detection model used to find the animal. |
| `hrnet_model` | `str` | `'hrnet_w32'` | The specific HRNet variant for pose estimation. |
| `max_individuals` | `int` | `1` | Maximum number of animals to track (currently supports 1). |
| `pcutoff` | `float` | `0.15` | Likelihood threshold for retaining keypoints. |
| `device` | `str` (optional) | `None` | 'cuda' or 'cpu'. Auto-detects if not specified. |

#### Usage Example

```python
from datafawn import DeepLabCutPoseEstimator

estimator = DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    device='cuda'
)

pose_data = estimator.estimate(
    video_path="my_video.mp4",
    output_path="output_folder"
)
```
