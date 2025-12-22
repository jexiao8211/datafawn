# Event Detection Pipeline

A modular, extensible pipeline system for pose estimation, postprocessing, and event extraction. Designed following SOLID principles with interchangeable components.

## Design Philosophy

### Why a Class-Based Approach?

A **class-based design** was chosen over a simple function because:

1. **State Management**: The pipeline maintains intermediate results (`pose_data`, `postprocessed_data`, `event_results`) that can be accessed after execution
2. **Configuration**: Pipeline components can be configured once and reused multiple times
3. **Extensibility**: Easy to add new steps, validators, or features without breaking existing code
4. **Composability**: Multiple pipelines can be created with different configurations
5. **Type Safety**: Abstract base classes provide clear interfaces and type hints

### Why Not Use Existing Packages?

While packages like `scikit-learn` have pipeline systems, they're designed for ML workflows with different requirements:
- **scikit-learn Pipeline**: Designed for transformers and estimators, not our use case
- **Luigi/Airflow**: Overkill for this application, designed for distributed workflows
- **Custom Solution**: Best fit for our specific needs (optional steps, multiple extractors, flexible data flow)

### Design Patterns Used

1. **Strategy Pattern**: Each step (pose estimation, postprocessing, event extraction) is an interchangeable strategy
2. **Template Method Pattern**: The pipeline defines the algorithm structure, but steps are customizable
3. **Dependency Injection**: Components are injected via constructor, making testing and swapping easy

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           EventDetectionPipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Pose Estimation (Optional)                    │
│  └─> PoseEstimator.estimate()                          │
│      └─> Returns: pd.DataFrame (pose data)             │
│                                                         │
│  Step 2: Postprocessing (Optional)                     │
│  └─> Postprocessor.process() (applied sequentially)    │
│      └─> Returns: pd.DataFrame (postprocessed data)   │
│                                                         │
│  Step 3: Event Extraction                              │
│  └─> EventExtractor.extract() (all run in parallel)    │
│      └─> Returns: Dict[str, Any] (events)             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Installation

Set up the conda environment:

```bash
conda env create -f environment.yml
conda activate fawn
```

## Usage

### Basic Example

```python
from event_detection import (
    EventDetectionPipeline,
    RelativePositionPostprocessor,
    ZeniExtractor
)

# Create pipeline
pipeline = EventDetectionPipeline(
    postprocessors=[RelativePositionPostprocessor()],
    event_extractors=[ZeniExtractor()]
)

# Run on existing pose data
results = pipeline.run(
    pose_data_path="processed_vids/deer2_pose.h5"
)

# Access results
strikes = results['events']['zeni']['strikes']
```

### With Video Input **WIP

```python
from event_detection import DeepLabCutPoseEstimator

# Create pose estimator
pose_estimator = DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    hrnet_model='hrnet_w32'
)

# Create pipeline
pipeline = EventDetectionPipeline(
    pose_estimator=pose_estimator,
    postprocessors=[RelativePositionPostprocessor()],
    event_extractors=[ZeniExtractor()]
)

# Run on video
results = pipeline.run(video_path="videos/deer2.mp4")
```

### Multiple Event Extractors

```python
# Create multiple extractors with different parameters
zeni_1 = ZeniExtractor(window_size=5, name="zeni_small_window")
zeni_2 = ZeniExtractor(window_size=7, name="zeni_large_window")

pipeline = EventDetectionPipeline(
    postprocessors=[RelativePositionPostprocessor()],
    event_extractors=[zeni_1, zeni_2]
)

results = pipeline.run(pose_data_path="data.h5")

# Access results from both
strikes_1 = results['events']['zeni_small_window']['strikes']
strikes_2 = results['events']['zeni_large_window']['strikes']
```

### Custom Parameters

```python
# Pass parameters with prefixes
results = pipeline.run(
    pose_data_path="data.h5",
    # Postprocessing parameters (prefix: postproc_)
    postproc_scorer=None,
    postproc_individual=None,
    # Event extraction parameters (prefix: extract_)
    extract_window_size=7,
    extract_prominence_percentage=0.03
)
```

### Complete Working Example

For a complete working example demonstrating the full pipeline setup, postprocessing, event extraction, and result access, see [`pipeline_test.ipynb`](pipeline_test.ipynb). This notebook shows:

- Setting up error detection and relative position postprocessors
- Configuring the Zeni event extractor
- Running the pipeline on existing pose data
- Accessing and working with the results

## Extending the Pipeline

### Creating a Custom Postprocessor

```python
from event_detection.pipeline import Postprocessor
import pandas as pd

class MyCustomPostprocessor(Postprocessor):
    def process(self, pose_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Your postprocessing logic here
        processed = pose_data.copy()
        # ... transformations ...
        return processed
```

### Creating a Custom Event Extractor

```python
from event_detection.pipeline import EventExtractor
import pandas as pd
from typing import Dict, Any

class MyCustomExtractor(EventExtractor):
    @property
    def name(self) -> str:
        return "my_custom_extractor"
    
    def extract(self, postprocessed_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Your event extraction logic here
        events = {...}
        return {
            'events': events,
            'metadata': {...}
        }
```

### Creating a Custom Pose Estimator

```python
from event_detection.pipeline import PoseEstimator
import pandas as pd
from pathlib import Path

class MyPoseEstimator(PoseEstimator):
    def estimate(self, video_path: Path, **kwargs) -> pd.DataFrame:
        # Your pose estimation logic here
        pose_data = ...  # Load or compute pose data
        return pose_data
```

## SOLID Principles Applied

1. **Single Responsibility**: Each class has one clear purpose
   - `PoseEstimator`: Only estimates pose
   - `Postprocessor`: Only postprocesses data
   - `EventExtractor`: Only extracts events
   - `EventDetectionPipeline`: Only orchestrates the pipeline

2. **Open/Closed**: Open for extension, closed for modification
   - Add new extractors without modifying the pipeline
   - Add new postprocessors without modifying existing code

3. **Liskov Substitution**: Any implementation can replace the base class
   - Any `PoseEstimator` can replace another
   - Any `Postprocessor` can replace another
   - Any `EventExtractor` can replace another

4. **Interface Segregation**: Small, focused interfaces
   - Each interface has only the methods it needs

5. **Dependency Inversion**: Depend on abstractions, not concretions
   - Pipeline depends on abstract base classes
   - Concrete implementations are injected

## Benefits

1. **Modularity**: Each step is independent and replaceable
2. **Testability**: Easy to mock components for testing
3. **Reusability**: Components can be reused in different pipelines
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new algorithms or steps
6. **Flexibility**: Support for optional steps and multiple extractors

## File Structure

```
event_detection/
├── __init__.py              # Package exports
├── pipeline.py              # Core pipeline and base classes
├── implementations.py       # Concrete implementations
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## Future Enhancements

Potential improvements:
- Caching intermediate results
- Parallel execution of extractors
- Pipeline validation and error handling
- Configuration file support
- Progress tracking and logging
- Pipeline visualization

