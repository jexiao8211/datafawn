# DataFawn: Event Detection Pipeline

A modular, extensible pipeline system for pose estimation, postprocessing, event extraction, and soundscape generation from animal movement videos. Designed following SOLID principles with interchangeable components.

## Overview

DataFawn transforms raw video footage into expressive, sonified visualizations by combining computer vision, signal processing, and audio synthesis. The system utilizes advanced pose estimation and event extraction techniques to create meaningful audio-visual representations of movement patterns.


https://github.com/user-attachments/assets/c3e8a886-3e0a-4f12-8ac8-0115d157ad0a


**Project Purpose:** DataFawn enables users to generate expressive videos that translate movement data into interesting compositions. By detecting and analyzing behavioral events in animal movement videos, the pipeline creates synchronized soundscapes where each movement event triggers corresponding notes, producing a fun representation of motion.

The pipeline processes video footage through four stages:
1. **Pose Estimation** - Track body parts using DeepLabCut SuperAnimal models
2. **Postprocessing** - Clean and normalize pose data (error correction, relative positioning)
3. **Event Extraction** - Detect specific behaviors (e.g., foot strikes)
4. **Soundscape Generation** - Create audio-visual compositions where events trigger musical notes

The pipeline is designed to be flexible: you can start from any stage if you already have intermediate results, making it easy to iterate on specific components.

## Features

- **Auto-detection**: Pipeline automatically determines which stages to run based on inputs
- **Flexible inputs**: Start from any stage (video, pose data, postprocessed data, or events)
- **Serialization**: Save and load intermediate results
- **Batch processing**: Process multiple videos at once
- **Error correction**: Multiple error detection methods (velocity, likelihood, distance)
- **Visualization**: Tools for plotting body part positions

## Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA-capable GPU (recommended for pose estimation, but not required)
- Windows, Linux, or macOS

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd datafawn
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate DEEPLABCUT
```

**Note**: The environment is named `DEEPLABCUT` (as specified in `environment.yml`). If you prefer a different name, modify the `name:` field in `environment.yml` before creating the environment.

3. Verify installation:
```python
import datafawn
print(datafawn.__version__ if hasattr(datafawn, '__version__') else "DataFawn installed")
```

## Quick Start

### Basic Example

```python
import datafawn
import torch

# Setup device (GPU recommended for pose estimation)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create pipeline components
pose_estimator = datafawn.DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    device=device
)

postprocessors = [
    datafawn.RelativePawPositionPostprocessor(),
    datafawn.ErrorPostprocessor(
        bodyparts=['front_left_paw', 'front_right_paw', 
                   'back_left_paw', 'back_right_paw'],
        use_velocity=True,
        use_likelihood=True
    )
]

event_extractor = datafawn.ZeniExtractor(
    smooth_window_size=5,
    prominence_percentage=0.05
)

# Create pipeline
pipeline = datafawn.EventDetectionPipeline(
    pose_estimator=pose_estimator,
    postprocessors=postprocessors,
    event_extractors=[event_extractor]
)

# Run full pipeline from video
results = pipeline.run(
    video_path="my_video.mp4",
    output_dir="results/my_video"
)

# Access results
print(f"Pose data shape: {results['pose_data'].shape}")
print(f"Events: {results['events']}")
print(f"Output files: {results['output_paths']}")
```

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
results = pipeline.run(
    events=my_events, 
    output_dir="output",
    soundscape_input_video="video.mp4"
)
```

## Output Directory Structure

When you provide `output_dir`, the pipeline creates an organized structure:

```
output_dir/
├── pose_estimation/
│   ├── pose_data.h5              # Raw pose estimation data
│   └── [labeled video files]     # DeepLabCut output files
├── postprocessing/
│   └── postprocessed_data.h5     # Postprocessed pose data
├── events/
│   └── events.json               # Extracted events
└── soundscapes/
    └── SoundScapeAuto_output.mp4  # Final video with audio
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

## Components

### Pose Estimators

#### DeepLabCutPoseEstimator

Uses DeepLabCut SuperAnimal models for pose estimation.

```python
estimator = datafawn.DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',  # Model type
    detector_name='fasterrcnn_resnet50_fpn_v2',  # Detector
    hrnet_model='hrnet_w32',  # HRNet variant
    max_individuals=1,  # Currently only supports 1
    pcutoff=0.15,  # Likelihood cutoff
    device=None  # Auto-detect or specify 'cuda'/'cpu'
)
```

**Supported Models:**
- `superanimal_quadruped` - For four-legged animals (deer, dogs, etc.)
- (WIP)

### Postprocessors

#### RelativePawPositionPostprocessor

Converts absolute paw coordinates to relative positions (relative to body reference points).

```python
rel_pp = datafawn.RelativePawPositionPostprocessor()
```

**Output:** Adds columns like `front_left_paw_rel`, `back_right_paw_rel`, etc.

#### ErrorPostprocessor

Detects and corrects errors in pose data using multiple methods.

```python
error_pp = datafawn.ErrorPostprocessor(
    bodyparts=['front_left_paw', 'front_right_paw', 
               'back_left_paw', 'back_right_paw'],
    use_velocity=True,  # Detect erroneous jumps in velocity
    use_likelihood=True,  # Detect low-confidence detections
    use_distance=True,  # Detect unrealistic proportions indicative of error
    velocity_kwargs={'threshold_pixels': 50, 'window_size': 5},
    likelihood_kwargs={'min_likelihood': 0.5},
    distance_kwargs={
        'reference_map': {
            'back_base': ['front_left_paw', 'front_right_paw'],
            'tail_base': ['back_left_paw', 'back_right_paw']
        },
        'max_distance': 300
    }
)
```

**Error Correction:** Detected errors are replaced with NaN, then filled using forward fill followed by backward fill.

### Event Extractors

#### ZeniExtractor

Detects foot strikes using the Zeni algorithm (simplified version).

```python
zeni = datafawn.ZeniExtractor(
    smooth_window_size=5,  # Smoothing window
    prominence_percentage=0.05,  # Peak detection threshold
    orientation_likelihood_threshold=0.0,  # Min likelihood for orientation
    orientation_smooth_window_size=15,  # Orientation smoothing
    show_plots=False,  # Display visualization plots
    name="zeni"  # Optional custom name
)
```

**Output Events:**
- `front_left_paw_strike`
- `front_right_paw_strike`
- `back_left_paw_strike`
- `back_right_paw_strike`

### Soundscape Generators

#### SoundScapeFromConfig

Manually map event types to specific sound files.

```python
config = {
    'event_sound_map': {
        'front_left_paw_strike': 'sounds/chime1.wav',
        'front_right_paw_strike': 'sounds/chime2.wav',
        'back_left_paw_strike': 'sounds/chime3.wav',
        'back_right_paw_strike': 'sounds/chime4.wav'
    },
    'backing_track': 'sounds/ambient.wav',  # Optional
    'volume': {  # Optional volume controls
        'backing_track': 0.5,
        'event_sounds': 1.0,
        'original_video': 0.3
    }
}

ss_gen = datafawn.SoundScapeFromConfig(soundscape_config=config)
```

#### SoundScapeAuto

Automatically selects notes based on movement speed. Higher speeds trigger higher notes.

```python
ss_auto = datafawn.SoundScapeAuto(
    notes_folder='sounds/custom_tone',  # Folder with note files (C5.wav, D5.wav, etc.)
    std_dev=1.5,  # Randomness in note selection
    speed_threshold=6.0,  # Speed threshold for reverse effect
    speed_window=60,  # Rolling window for speed calculation (frames)
    backing_track_path='sounds/calm_ambient_backing.wav',  # Optional
    backing_track_base_volume=0.5,  # Min volume
    backing_track_max_volume=1.0,  # Max volume at threshold
    backing_track_volume_curve=3.0,  # Volume scaling curve (cubic)
    show_speed_plot=False,  # Display speed plot
    speed_plot_path=None  # Save speed plot to file
)
```

**Features:**
- Front feet sample from higher octave (C6-G7) than back feet (C5-G6)
- Speed-based note selection using normal distribution
- Audio reversal when speed exceeds threshold
- Dynamic backing track volume scaling with speed

**Note Files Required:**
- Back feet: C4-B6 (major scale notes)
- Front feet: C5-B7 (major scale notes)
- Files should be named like `C5.wav`, `D5.wav`, etc.

## Serialization: Save and Load Results

### Save Results

```python
results = pipeline.run(video_path="video.mp4", output_dir="output")

# Save all results to a directory
saved_paths = pipeline.save_results(results, "saved_results")
```

### Load Results Later

```python
# Load previously saved results
loaded = datafawn.EventDetectionPipeline.load_results("saved_results")

# Continue processing from loaded data
results = pipeline.run(
    events=loaded['events'],
    soundscape_input_video="video.mp4",
    output_dir="output"
)
```

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

## Visualization

### Plot Body Part Positions

```python
import datafawn

# Plot X and Y positions over time
fig = datafawn.plot_bodyparts_position(
    pose_data=results['pose_data'],
    bodyparts=['front_left_paw', 'back_base'],
    scorer='superanimal_quadruped_hrnet_w32',
    individual='animal0',
    min_likelihood=0.5,
    figsize=(15, 5)
)
```

## Complete Example

```python
import datafawn
import torch
from pathlib import Path

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create components
dlc_estimator = datafawn.DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    device=device
)

rel_pp = datafawn.RelativePawPositionPostprocessor()

error_pp = datafawn.ErrorPostprocessor(
    bodyparts=['front_left_paw', 'front_right_paw', 
               'back_left_paw', 'back_right_paw'],
    use_velocity=True,
    use_likelihood=True,
    velocity_kwargs={'threshold_pixels': 50},
    likelihood_kwargs={'min_likelihood': 0.5}
)

zeni_extractor = datafawn.ZeniExtractor(
    smooth_window_size=5,
    prominence_percentage=0.05
)

# Option 1: Manual soundscape configuration
soundscape_config = {
    'event_sound_map': {
        'front_left_paw_strike': 'sounds/chime1.wav',
        'front_right_paw_strike': 'sounds/chime2.wav',
        'back_left_paw_strike': 'sounds/chime3.wav',
        'back_right_paw_strike': 'sounds/chime4.wav'
    }
}
ss_gen = datafawn.SoundScapeFromConfig(soundscape_config=soundscape_config)

# Option 2: Automatic speed-based soundscape
ss_auto = datafawn.SoundScapeAuto(
    notes_folder='sounds/custom_tone',
    speed_threshold=6.0,
    backing_track_path='sounds/calm_ambient_backing.wav'
)

# Create pipeline
pipeline = datafawn.EventDetectionPipeline(
    pose_estimator=dlc_estimator,
    postprocessors=[rel_pp, error_pp],
    event_extractors=[zeni_extractor],
    soundscape_generators=[ss_auto]  # or [ss_gen]
)

# Run pipeline
results = pipeline.run(
    video_path="my_video.mp4",
    output_dir="results/my_video",
    soundscape_input_video="pose_est"  # Use labeled video from pose estimation
)

# Check results
print(f"Pose data shape: {results['pose_data'].shape}")
print(f"Postprocessed shape: {results['postprocessed_data'].shape}")
print(f"Events extracted: {len(results['events'])} individuals")
print(f"Output files: {list(results['output_paths'].keys())}")
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
import pandas as pd

class MyPoseEstimator(PoseEstimator):
    def estimate(self, video_path: Path, output_path=None, **kwargs) -> pd.DataFrame:
        # Your pose estimation logic
        return pose_data
```

### Custom Soundscape Generator

```python
from datafawn.pipeline import SoundScapeGenerator
from typing import Dict, Any, Optional, Union
from pathlib import Path

class MySoundScapeGenerator(SoundScapeGenerator):
    def generate(
        self, 
        input_video_path: Union[str, Path], 
        events: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        # Your soundscape generation logic
        return str(output_path)
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
├── __init__.py                    # Package exports
├── pipeline.py                    # Core pipeline and base classes
├── pose_estimators.py             # Pose estimation implementations
├── postprocessors.py              # Postprocessing implementations
├── event_extractors.py            # Event extraction implementations
├── soundscape_gen.py              # Soundscape generation implementations
├── vis.py                         # Visualization utilities
├── event_extraction/
│   └── zeni.py                    # Zeni algorithm implementation
├── postprocessing/
│   ├── error_mask.py              # Error detection functions
│   └── norm.py                   # Normalization utilities
└── soundscape/
    ├── __init__.py
    ├── soundscape_auto.py         # Automatic soundscape generation
    ├── soundscape_from_config.py  # Config-based soundscape generation
    ├── audio_utils.py             # Audio processing utilities
    └── event_utils.py             # Event analysis utilities
```

## Demo Notebooks

- [`pipeline_demo.ipynb`](pipeline_demo.ipynb) - Quick demo of pipeline features
- [`pipeline_full_demo.ipynb`](pipeline_full_demo.ipynb) - Comprehensive examples with all stages
- [`soundscape_building.ipynb`](soundscape_building.ipynb) - Soundscape generation examples
- [`batch_process.ipynb`](batch_process.ipynb) - Batch processing examples

## Requirements

### System Requirements

**DeepLabCut Requirements** (for pose estimation):
- **Python**: 3.7-3.10 (3.10 recommended, 3.12 supported via this environment)
- **GPU**: NVIDIA GPU with CUDA support (highly recommended for pose estimation)
  - CUDA 11.0 or higher
  - Minimum 4GB VRAM, 8GB+ recommended
- **RAM**: 8GB minimum, 16GB+ recommended for multi-animal tracking
- **Storage**: At least 10GB free disk space for software and pre-trained models

**Additional Requirements** (for other pipeline stages):
- **RAM**: 4GB minimum (postprocessing, event extraction, and soundscape generation are lightweight)
- **Storage**: Sufficient space for video files and outputs (varies by project size)

### Key Dependencies

- Python 3.12
- DeepLabCut 3.0.0rc13
- PyTorch 2.9.1 (with CUDA support)
- MoviePy 2.2.1
- Pandas 3.0.0rc0
- NumPy 1.26.4
- SciPy 1.17.0rc1
- Matplotlib 3.8.4

See `environment.yml` for complete dependency list.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Missing note files**: Ensure all required note files exist in `notes_folder`
3. **Video codec issues**: Ensure ffmpeg is properly installed
4. **Import errors**: Make sure conda environment is activated

### Getting Help

- Check the demo notebooks for examples
- Review the docstrings in the code
- Ensure all dependencies are installed correctly

## Contributing

Contributions are welcome! When extending the pipeline:

1. Follow the existing architecture patterns
2. Implement the appropriate base class interface
3. Add docstrings following the existing style
4. Test with the demo notebooks

## Contact

For questions, support, or collaboration inquiries:

- **Email**: jexiao8211@gmail.com

## Acknowledgments

- DeepLabCut team for pose estimation models
- Zeni et al. for the foot strike detection algorithm

---

**Note**: This project is designed for research and creative applications. Ensure you have appropriate permissions for any video content you process.
