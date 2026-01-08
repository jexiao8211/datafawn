# Postprocessing

Postprocessing cleans and transforms the raw pose data to make it suitable for event extraction. This can include error correction and calculating relative positions.

## Classes

### `ErrorPostprocessor`

Detects and corrects errors in the pose estimation data. It can use velocity, likelihood, and geometric distance to identify unlikely poses.

**Location:** `datafawn.postprocessors`

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bodyparts` | `List[str]` | (Required) | List of body parts to monitor for errors. |
| `use_velocity` | `bool` | `True` | Check for unrealistic jumps in position between frames. |
| `use_likelihood` | `bool` | `True` | Filter out points with low confidence scores. |
| `use_distance` | `bool` | `True` | Check for unrealistic distances between body parts. |
| `velocity_kwargs` | `dict` | `{}` | Config for velocity check (e.g., `threshold_pixels`). |
| `likelihood_kwargs` | `dict` | `{}` | Config for likelihood check (e.g., `min_likelihood`). |
| `distance_kwargs` | `dict` | `{}` | Config for distance check (requires `reference_map`). |

#### Usage Example

```python
from datafawn import ErrorPostprocessor

error_pp = ErrorPostprocessor(
    bodyparts=['front_left_paw', 'front_right_paw'],
    use_velocity=True,
    velocity_kwargs={'threshold_pixels': 50}
)
```

### `RelativePawPositionPostprocessor`

Calculates the position of paws relative to other body parts (like the hips or shoulders) rather than absolute screen coordinates. This is often robust for detecting events like foot strikes.

**Location:** `datafawn.postprocessors`

#### Initialization Parameters

This class takes no initialization parameters.

#### Usage Example

```python
from datafawn import RelativePawPositionPostprocessor

rel_pp = RelativePawPositionPostprocessor()
```
