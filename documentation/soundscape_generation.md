# Soundscape Generation

Soundscape generation takes the extracted events and the original video to create a final video with an audio overlay.

## Classes

### `SoundScapeAuto`

Automatically selects musical notes based on the speed of movement. Faster movement triggers higher notes, and movement exceeding a threshold can trigger special effects like audio reversal.

**Location:** `datafawn.soundscape_gen`

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `notes_folder` | `str` or `Path` | `"sounds/custom_tone"` | Directory containing note files (e.g., C5.wav). |
| `std_dev` | `float` | `1.5` | Randomness in note selection around the target pitch. |
| `speed_threshold` | `float` | `6.0` | Speed threshold (footfalls/window) for max volume/reversal. |
| `speed_window` | `int` | `60` | Rolling window size (frames) for calculating speed. |
| `backing_track_path` | `str` (optional) | `None` | Path to a background audio track. |
| `backing_track_base_volume` | `float` | `1.0` | Minimum volume for the backing track. |
| `backing_track_max_volume` | `float` | `1.0` | Maximum volume for the backing track at high speeds. |

#### Usage Example

```python
from datafawn import SoundScapeAuto

auto_gen = SoundScapeAuto(
    notes_folder='sounds/piano',
    speed_threshold=8.0
)
```

### `SoundScapeFromConfig`

Maps specific event types to specific sound files using a configuration dictionary. This allows for manual control over which sound plays for each event.

**Location:** `datafawn.soundscape_gen`

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `soundscape_config` | `dict` | (Required) | Dictionary defining the mapping between events and sounds. |

**Config Structure:**
```python
config = {
    'event_sound_map': {
        'front_left_paw_strike': 'sounds/bell.wav',
        'back_right_paw_strike': 'sounds/kick.wav'
    },
    'backing_track': 'sounds/ambient.mp3',
    'volume': {
        'backing_track': 0.5,
        'event_sounds': 1.0,
        'original_video': 0.0
    }
}
```

#### Usage Example

```python
from datafawn import SoundScapeFromConfig

manual_gen = SoundScapeFromConfig(soundscape_config=my_config)
```
