# Event Detection Pipeline

A modular, extensible pipeline system for pose estimation, postprocessing, event extraction, and soundscape generation. Designed following SOLID principles with interchangeable components.

## Features

- **Auto-detection**: Pipeline automatically determines which stages to run based on inputs
- **Flexible inputs**: Start from any stage (video, pose data, postprocessed data, or events)
- **Organized outputs**: Clear directory structure for all outputs
- **Serialization**: Save and load intermediate results
- **Batch processing**: Process multiple videos at once

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               EventDetectionPipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Pose Estimation (skipped if pose_data provided)      │
│  └─> PoseEstimator.estimate()                                  │
│      └─> Returns: pd.DataFrame (pose data)                     │
│                                                                 │
│  Stage 2: Postprocessing (skipped if postprocessed_data)       │
│  └─> Postprocessor.process() (applied sequentially)            │
│      └─> Returns: pd.DataFrame (postprocessed data)            │
│                                                                 │
│  Stage 3: Event Extraction (skipped if events provided)        │
│  └─> EventExtractor.extract() (all run in parallel)            │
│      └─> Returns: Dict[str, Any] (events)                      │
│                                                                 │
│  Stage 4: Soundscape Generation                                │
│  └─> SoundScapeGenerator.generate()                            │
│      └─> Returns: video with audio overlay                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Details

| Stage | Method | Input | Output | Data Structure |
|-------|--------|-------|--------|----------------|
| **1. Pose Estimation** | `PoseEstimator.estimate()` | `video_path`, `output_path` | `pd.DataFrame` | MultiIndex columns: (scorer, individual, bodypart, coords) |
| **2. Postprocessing** | `Postprocessor.process()` | `pd.DataFrame` | `pd.DataFrame` | Same structure, with added columns (e.g., `_rel` positions) |
| **3. Event Extraction** | `EventExtractor.extract()` | `pd.DataFrame` | `Dict` | `{'events': {(scorer, individual): {event_type: [frames]}}, 'metadata': {...}}` |
| **4. Soundscape Gen** | `SoundScapeGenerator.generate()` | `video_path`, `events`, `output_path` | `str` | Path to output video |

**Events Structure Example:**
```python
{
    ('superanimal_quadruped_hrnet_w32', 'animal0'): {
        'front_left_paw_strike': [47, 79, 122, 154],
        'front_right_paw_strike': [45, 77, 104, 139],
        'back_left_paw_strike': [111, 144, 175, 206],
        'back_right_paw_strike': [100, 133, 165, 195]
    }
}
```

## Installation

Set up the conda environment:

```bash
conda env create -f environment.yml
conda activate fawn
```

## Quick Start

```python
import datafawn

# Create pipeline with all components
pipeline = datafawn.EventDetectionPipeline(
    pose_estimator=datafawn.DeepLabCutPoseEstimator(),
    postprocessors=[datafawn.RelativePawPositionPostprocessor()],
    event_extractors=[datafawn.ZeniExtractor()],
    soundscape_generators=[datafawn.SoundScapeFromConfig(config)]
)

# Run full pipeline from video
results = pipeline.run(
    video_path="my_video.mp4",
    output_dir="results/my_video"
)
```

## Auto-Detection: Start from Any Stage

The pipeline automatically detects which stages to run based on your inputs:

| Input Provided | Stages Run |
|----------------|------------|
| `video_path` | Pose Est → Postproc → Events → Soundscape |
| `pose_data` or `pose_data_path` | Postproc → Events → Soundscape |
| `postprocessed_data` or `postprocessed_data_path` | Events → Soundscape |
| `events` or `events_path` | Soundscape only |

### Examples

```python
# Full pipeline from video
results = pipeline.run(video_path="video.mp4", output_dir="output")

# Skip pose estimation (use existing pose data)
results = pipeline.run(pose_data_path="pose_data.h5", output_dir="output")

# Skip pose estimation + postprocessing
results = pipeline.run(postprocessed_data=my_dataframe, output_dir="output")

# Only generate soundscape (use existing events)
results = pipeline.run(events=my_events, output_dir="output")
```

## Output Directory Structure

When you provide `output_dir`, the pipeline creates an organized structure:

```
output_dir/
├── pose_estimation/
│   └── pose_data.h5              # Raw pose estimation data
├── postprocessing/
│   └── postprocessed_data.h5     # Postprocessed pose data
├── events/
│   └── events.json               # Extracted events
└── soundscapes/
    └── SoundScapeFromConfig_output.mp4  # Final video with audio
```

For batch processing, each video gets its own subdirectory:

```
batch_output/
├── video1/
│   ├── pose_estimation/
│   ├── postprocessing/
│   ├── events/
│   └── soundscapes/
├── video2/
│   └── ...
└── video3/
    └── ...
```

## Serialization: Save and Load Results

### Save results
```python
results = pipeline.run(video_path="video.mp4", output_dir="output")

# Save all results to a directory
saved_paths = pipeline.save_results(results, "saved_results")
```

### Load results later
```python
# Load previously saved results
loaded = datafawn.EventDetectionPipeline.load_results("saved_results")

# Continue processing from loaded data
results = pipeline.run(
    events=loaded['events'],
    soundscape_input_video="video.mp4"
)
```

## Soundscape Configuration

The `SoundScapeFromConfig` generator requires a configuration dictionary that maps event types to sound files.

### Configuration Format

```python
soundscape_config = {
    'event_sound_map': {
        'event_type_1': 'path/to/sound1.wav',
        'event_type_2': 'path/to/sound2.wav',
        # ... more event types
    }
}
```

### Requirements

1. **`event_sound_map`** (required): Dictionary mapping event type names to sound file paths
   - Keys must match the event type names from your event extractor (e.g., `'front_left_paw_strike'`)
   - Values can be `str` or `Path` objects pointing to audio files (`.wav`, `.mp3`, etc.)
   - Sound files must exist at the specified paths

2. **Event Type Names**: Must exactly match the event types in your extracted events
   - Example event types from `ZeniExtractor`: 
     - `'front_left_paw_strike'`
     - `'front_right_paw_strike'`
     - `'back_left_paw_strike'`
     - `'back_right_paw_strike'`

### Example

```python
from pathlib import Path

soundscape_config = {
    'event_sound_map': {
        'front_left_paw_strike': Path('sounds/chime1.wav'),
        'front_right_paw_strike': Path('sounds/chime2.wav'),
        'back_left_paw_strike': Path('sounds/chime3.wav'),
        'back_right_paw_strike': Path('sounds/chime4.wav')
    }
}

ss_generator = datafawn.SoundScapeFromConfig(soundscape_config=soundscape_config)
```

**Note**: If an event type in your events doesn't have a corresponding entry in `event_sound_map`, that event type will be skipped with a warning. If an event type is in the config but not found in the events, it will also be skipped.

## Batch Processing

Process multiple videos at once:

```python
results_list = pipeline.run_batch(
    video_paths=["vid1.mp4", "vid2.mp4", "vid3.mp4"],
    output_base_dir="batch_results"
)

# Each result contains success status
for result in results_list:
    if result['success']:
        print(f"✅ {result['video_path'].name}: {len(result['events'])} events")
    else:
        print(f"❌ {result['video_path'].name}: {result['error']}")
```

## Complete Example

```python
import datafawn
import torch

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create components
dlc_estimator = datafawn.DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    device=device
)

rel_pp = datafawn.RelativePawPositionPostprocessor()
error_pp = datafawn.ErrorPostprocessor(
    bodyparts=['front_left_paw_rel', 'front_right_paw_rel', 
               'back_left_paw_rel', 'back_right_paw_rel'],
    use_velocity=True,
    use_likelihood=True
)

zeni_extractor = datafawn.ZeniExtractor(window_size=5)

soundscape_config = {
    'event_sound_map': {
        'front_left_paw_strike': 'sounds/chime1.wav',
        'front_right_paw_strike': 'sounds/chime2.wav',
        'back_left_paw_strike': 'sounds/chime3.wav',
        'back_right_paw_strike': 'sounds/chime4.wav'
    }
}
ss_gen = datafawn.SoundScapeFromConfig(soundscape_config)

# Create pipeline
pipeline = datafawn.EventDetectionPipeline(
    pose_estimator=dlc_estimator,
    postprocessors=[rel_pp, error_pp],
    event_extractors=[zeni_extractor],
    soundscape_generators=[ss_gen]
)

# Run pipeline
results = pipeline.run(
    video_path="my_video.mp4",
    output_dir="results/my_video",
    soundscape_input_video="pose_est"  # Use labeled video for soundscape
)

# Check results
print(f"Pose data shape: {results['pose_data'].shape}")
print(f"Events: {results['events']}")
print(f"Output files: {results['output_paths']}")
```

## Design Philosophy

### Why a Class-Based Approach?

1. **State Management**: Pipeline maintains intermediate results accessible after execution
2. **Configuration**: Components can be configured once and reused multiple times
3. **Extensibility**: Easy to add new steps without breaking existing code
4. **Composability**: Multiple pipelines can be created with different configurations

### Design Patterns Used

1. **Strategy Pattern**: Each step is an interchangeable strategy
2. **Template Method Pattern**: Pipeline defines the algorithm structure, steps are customizable
3. **Dependency Injection**: Components are injected via constructor

## Extending the Pipeline

### Custom Postprocessor

```python
from datafawn.pipeline import Postprocessor
import pandas as pd

class MyPostprocessor(Postprocessor):
    def process(self, pose_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        processed = pose_data.copy()
        # Your logic here
        return processed
```

### Custom Event Extractor

```python
from datafawn.pipeline import EventExtractor
from typing import Dict, Any

class MyExtractor(EventExtractor):
    @property
    def name(self) -> str:
        return "my_extractor"
    
    def extract(self, postprocessed_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        events = {}  # Your extraction logic
        return {'events': events, 'metadata': {}}
```

### Custom Pose Estimator

```python
from datafawn.pipeline import PoseEstimator
from pathlib import Path

class MyPoseEstimator(PoseEstimator):
    def estimate(self, video_path: Path, **kwargs) -> pd.DataFrame:
        # Your pose estimation logic
        return pose_data
```

## SOLID Principles Applied

1. **Single Responsibility**: Each class has one clear purpose
2. **Open/Closed**: Open for extension, closed for modification
3. **Liskov Substitution**: Any implementation can replace the base class
4. **Interface Segregation**: Small, focused interfaces
5. **Dependency Inversion**: Depend on abstractions, not concretions

## File Structure

```
datafawn/
├── __init__.py              # Package exports
├── pipeline.py              # Core pipeline and base classes
├── pose_estimators.py       # Pose estimation implementations
├── postprocessors.py        # Postprocessing implementations
├── event_extractors.py      # Event extraction implementations
└── soundscape_gen.py        # Soundscape generation
```

## Demo Notebooks

- [`pipeline_demo.ipynb`](pipeline_demo.ipynb) - Quick demo of new features
- [`pipeline_full_demo.ipynb`](pipeline_full_demo.ipynb) - Full pipeline with all stages
- [`pipeline_test.ipynb`](pipeline_test.ipynb) - Testing and development
