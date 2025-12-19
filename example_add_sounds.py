"""
Example script showing how to use add_sounds_to_video.py with zeni algorithm output.

This demonstrates the complete workflow:
1. Run zeni algorithm to get strikes
2. Add sounds to video based on strikes
"""

from pathlib import Path
import pandas as pd
from utils.zeni import zeni_algorithm
from utils.postprocessing import paw_to_relative_position, detect_pose_errors
from add_sounds_to_video import add_sounds_to_video, save_strikes_to_json, load_strikes_from_json

# Configuration
VIDEO_PATH = Path('videos/deer2.mp4')
H5_FILE = 'processed_vids/deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5'
SOUND_FILE = Path('chime.wav')  # You need to provide a sound file
PAW_TO_TRACK = 'front_right_paw'  # Options: 'front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw'
OUTPUT_VIDEO = Path('videos/deer2_with_chimes.mp4')
STRIKES_JSON = 'strikes_output.json'  # Optional: save strikes for later use

# Step 1: Load pose data
print("Loading pose data...")
pose_data = pd.read_hdf(H5_FILE)
pose_data = pose_data.sort_index(axis=1)

# Step 2: Get error masks
print("Detecting pose errors...")
paws = ['front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw']
error_details = detect_pose_errors(
    pose_data,
    bodyparts=paws,
    velocity_threshold=50,
    min_likelihood=0.0,
    max_distance=300
)

error_mask = error_details[['front_left_paw_error', 'front_right_paw_error', 
                            'back_left_paw_error', 'back_right_paw_error']].copy()
error_mask.columns = [col.replace('_error', '') for col in error_mask.columns]

# Step 3: Calculate relative positions
print("Calculating relative positions...")
pose_data_with_rel = paw_to_relative_position(pose_data, append_to_df=True)

# Step 4: Run zeni algorithm
print("Running zeni algorithm...")
strikes = zeni_algorithm(
    pose_data_with_rel,
    window_size=5,
    min_contact_duration=3,
    velocity_threshold=10,
    error_mask=error_mask
)

print(f"\nStrikes detected:")
for paw, frames in strikes.items():
    print(f"  {paw}: {len(frames)} strikes at frames {frames[:5]}..." if len(frames) > 5 else f"  {paw}: {len(frames)} strikes at frames {frames}")

# Step 5: (Optional) Save strikes to JSON for later use
save_strikes_to_json(strikes, STRIKES_JSON)
print(f"\nStrikes saved to {STRIKES_JSON}")

# Step 6: Add sounds to video
# Option A: Add sounds for a single paw
print(f"\n=== Option A: Single Paw ===")
print(f"Adding sounds to video for {PAW_TO_TRACK}...")
print(f"Using sound file: {SOUND_FILE}")

if not SOUND_FILE.exists():
    print(f"\nWARNING: Sound file not found at {SOUND_FILE}")
    print("Please provide a sound file (e.g., chime.wav) to use this feature.")
    print("You can download free sound effects from sites like:")
    print("  - https://freesound.org/")
    print("  - https://www.zapsplat.com/")
else:
    from add_sounds_to_video import add_sounds_to_video
    output_path = add_sounds_to_video(
        video_path=VIDEO_PATH,
        strikes_dict=strikes,
        sound_path=SOUND_FILE,
        paw_name=PAW_TO_TRACK,
        output_path=OUTPUT_VIDEO,
        fps=29.97  # From your video metadata
    )
    print(f"\n✓ Success! Video with sounds saved to: {output_path}")

# Option B: Add different sounds for all paws simultaneously
print(f"\n=== Option B: All Paws at Once ===")
print("Adding different sounds for each paw in a single video...")

# Define different sounds for each paw (you can use the same sound for all if you want)
paw_sound_map = {
    'front_left_paw': 'sound1.wav',   # Replace with your sound files
    'front_right_paw': 'sound2.wav',
    'back_left_paw': 'sound3.wav',
    'back_right_paw': 'sound4.wav'
}

# Or use the same sound for all paws:
# paw_sound_map = {
#     'front_left_paw': 'chime.wav',
#     'front_right_paw': 'chime.wav',
#     'back_left_paw': 'chime.wav',
#     'back_right_paw': 'chime.wav'
# }

# Check if all sound files exist
missing_sounds = [paw for paw, sound in paw_sound_map.items() 
                  if not Path(sound).exists()]

if missing_sounds:
    print(f"\nWARNING: Some sound files not found:")
    for paw in missing_sounds:
        print(f"  {paw}: {paw_sound_map[paw]}")
    print("\nPlease provide sound files to use this feature.")
else:
    from add_sounds_to_video import add_sounds_for_all_paws
    output_path_all = add_sounds_for_all_paws(
        video_path=VIDEO_PATH,
        strikes_dict=strikes,
        paw_sound_map=paw_sound_map,
        output_path=VIDEO_PATH.parent / f"{VIDEO_PATH.stem}_all_paws_sounds.mp4",
        fps=29.97
    )
    print(f"\n✓ Success! Video with all paw sounds saved to: {output_path_all}")

# Alternative: Load strikes from JSON if you've already run zeni algorithm
# strikes = load_strikes_from_json(STRIKES_JSON)

