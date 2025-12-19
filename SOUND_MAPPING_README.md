# Adding Sounds to Video Based on Paw Strikes

This guide explains how to add sounds (like chimes) to a video every time a specific paw hits the ground, using the output from the zeni algorithm.

## Overview

The `add_sounds_to_video.py` script takes:
- A video file (.mp4)
- A strikes dictionary from `zeni_algorithm()` (frame indices where paws hit ground)
- A sound file (.wav, .mp3, etc.)
- Which paw to track (e.g., 'front_right_paw')

And creates a new video with the sound playing at each strike timestamp.

## Installation

You'll need to install `moviepy`:

```bash
pip install moviepy
```

Or if using conda:

```bash
conda install -c conda-forge moviepy
```

## Quick Start

### Option 1: Add sounds for all paws at once (Recommended)

This creates a single video with different sounds for each paw:

```python
from add_sounds_to_video import add_sounds_for_all_paws

# Define different sounds for each paw
paw_sound_map = {
    'front_left_paw': 'chime1.wav',
    'front_right_paw': 'chime2.wav',
    'back_left_paw': 'thud.wav',
    'back_right_paw': 'click.wav'
}

# Or use the same sound for all paws:
# paw_sound_map = {
#     'front_left_paw': 'chime.wav',
#     'front_right_paw': 'chime.wav',
#     'back_left_paw': 'chime.wav',
#     'back_right_paw': 'chime.wav'
# }

output_path = add_sounds_for_all_paws(
    video_path='videos/deer2.mp4',
    strikes_dict=strikes,  # From zeni_algorithm
    paw_sound_map=paw_sound_map,
    fps=29.97
)
```

### Option 2: Add sounds for a single paw

If you only want sounds for one paw:

```python
from add_sounds_to_video import add_sounds_to_video

output_path = add_sounds_to_video(
    video_path='videos/deer2.mp4',
    strikes_dict=strikes,
    sound_path='chime.wav',
    paw_name='front_right_paw',
    fps=29.97
)
```

### Option 3: Use the example script

1. Make sure you have sound files in your project directory
2. Run the example script:

```python
python example_add_sounds.py
```

This will:
- Load your pose data
- Run the zeni algorithm to detect strikes
- Show examples of both single-paw and all-paws approaches

## Function Reference

### `add_sounds_for_all_paws()` ⭐ Recommended

Add different sounds for each paw simultaneously in a single video.

**Parameters:**
- `video_path` (str or Path): Path to input video file
- `strikes_dict` (dict): Dictionary from zeni_algorithm with paw names as keys and frame lists as values
- `paw_sound_map` (dict): Dictionary mapping paw names to sound file paths. Example:
  ```python
  {
      'front_left_paw': 'sound1.wav',
      'front_right_paw': 'sound2.wav',
      'back_left_paw': 'sound3.wav',
      'back_right_paw': 'sound4.wav'
  }
  ```
  You can omit paws you don't want sounds for.
- `output_path` (str or Path, optional): Output video path. If None, auto-generates name.
- `fps` (float, optional): Video FPS. If None, reads from video file.

**Returns:**
- Path to output video file

**Example:**
```python
strikes = {
    'front_right_paw': [180, 205, 270, 298, 316, 335, 350, 366, 392, 485],
    'front_left_paw': [88, 132, 163, 204, 223, 254, 287, 304, 325, 344, 363, 385, 483, 511],
    'back_left_paw': [120, 160, 180, 224, 251, 277, 327, 346, 365, 386],
    'back_right_paw': [116, 143, 166, 192, 224, 253, 280, 306, 325, 344, 356, 376, 385, 495]
}

# Different sounds for each paw
paw_sound_map = {
    'front_left_paw': 'chime_high.wav',
    'front_right_paw': 'chime_medium.wav',
    'back_left_paw': 'thud.wav',
    'back_right_paw': 'click.wav'
}

output = add_sounds_for_all_paws(
    video_path='videos/deer2.mp4',
    strikes_dict=strikes,
    paw_sound_map=paw_sound_map
)
```

### `add_sounds_to_video()`

Add sounds to video for a single paw.

**Parameters:**
- `video_path` (str or Path): Path to input video file
- `strikes_dict` (dict): Dictionary from zeni_algorithm with paw names as keys and frame lists as values
- `sound_path` (str or Path): Path to sound file (.wav, .mp3, etc.)
- `paw_name` (str): Which paw to track. Options:
  - `'front_left_paw'`
  - `'front_right_paw'`
  - `'back_left_paw'`
  - `'back_right_paw'`
- `output_path` (str or Path, optional): Output video path. If None, auto-generates name.
- `fps` (float, optional): Video FPS. If None, reads from video file.

**Returns:**
- Path to output video file

**Example:**
```python
output = add_sounds_to_video(
    video_path='videos/deer2.mp4',
    strikes_dict=strikes,
    sound_path='chime.wav',
    paw_name='front_right_paw'
)
```

### Helper Functions

**`save_strikes_to_json(strikes_dict, json_path)`**
- Save strikes dictionary to JSON file for later use

**`load_strikes_from_json(json_path)`**
- Load strikes dictionary from JSON file

## Getting Sound Files

You can download free sound effects from:
- [Freesound.org](https://freesound.org/)
- [Zapsplat](https://www.zapsplat.com/)
- [Pixabay](https://pixabay.com/sound-effects/)

Or create your own using tools like Audacity.

## Notes

- The script preserves the original video audio and overlays the strike sounds
- If a strike timestamp is beyond the video duration, that sound is skipped
- Frame numbers are converted to timestamps using: `timestamp = frame_number / fps`
- The output video uses H.264 codec for video and AAC for audio (widely compatible)

## Troubleshooting

**"Sound file not found"**
- Make sure the sound file path is correct
- Use absolute paths if relative paths don't work

**"Paw not found in strikes_dict"**
- Check that you're using the correct paw name
- Available names: `'front_left_paw'`, `'front_right_paw'`, `'back_left_paw'`, `'back_right_paw'`

**Video processing is slow**
- This is normal - video encoding takes time
- The script will show progress

**No sounds in output**
- Check that strikes were detected for the paw you selected
- Verify the sound file plays correctly
- Check that timestamps are within video duration

