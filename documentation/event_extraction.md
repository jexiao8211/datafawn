# Event Extraction

Event extraction identifies specific behaviors or occurrences from the postprocessed pose data, such as foot strikes (steps).

## Classes

### `ZeniExtractor`

Implements a simplified version of the Zeni algorithm to detect foot strikes. It looks for peaks and troughs in the movement data to identify when a paw hits the ground.

**Location:** `datafawn.event_extractors`

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smooth_window_size` | `int` | `5` | Size of the window used to smooth the signal before detection. |
| `prominence_percentage` | `float` | `0.05` | Minimum height of a peak relative to neighbors to be counted. |
| `orientation_likelihood_threshold` | `float` | `0.0` | Minimum confidence required for orientation data (if used). |
| `orientation_smooth_window_size` | `int` | `15` | Window size for smoothing orientation data. |
| `show_plots` | `bool` | `False` | If True, generates plots visualizing the detection process. |
| `name` | `str` | `"zeni"` | A unique name for this extractor instance. |

#### Usage Example

```python
from datafawn import ZeniExtractor

extractor = ZeniExtractor(
    smooth_window_size=5,
    prominence_percentage=0.1
)

# Returns dictionary with events like 'front_left_paw_strike': [frames...]
events = extractor.extract(postprocessed_data)
```
