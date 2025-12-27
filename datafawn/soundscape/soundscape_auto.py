"""
Implementation of automatic soundscape generation with speed-based note selection.

This module generates soundscapes by automatically selecting notes from a major
scale based on the relative speed of movement at each event frame. Front feet
sample from notes one octave higher than back feet.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip

def audio_time_mirror(clip, file_path=None):
    """
    Reverse audio clip by flipping the audio array.
    
    Parameters
    ----------
    clip : AudioFileClip
        The audio clip to reverse
    file_path : str or Path, optional
        Path to the audio file. If provided, reads directly from file
        to bypass moviepy's reader issues with short clips.
    
    Returns
    -------
    AudioArrayClip
        A new reversed audio clip
    """
    fps = clip.fps
    
    # Read the WAV file directly using scipy to bypass moviepy's reader issues
    # This is more reliable for short clips and avoids the "t=1.00-1.00" error
    if file_path is not None:
        try:
            from scipy.io import wavfile
            sample_rate, audio_array = wavfile.read(str(file_path))
            
            # Convert to float32 if needed (wavfile.read returns int16 for most files)
            if audio_array.dtype != np.float32:
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    audio_array = audio_array.astype(np.float32)
            
            # Handle mono vs stereo
            if len(audio_array.shape) == 1:
                # Mono: add channel dimension
                audio_array = audio_array[:, np.newaxis]
            
            # Reverse along time axis (axis 0)
            reversed_array = np.flip(audio_array, axis=0).copy()
            
            # Create AudioArrayClip using the original clip's fps
            from moviepy.audio.AudioClip import AudioArrayClip
            return AudioArrayClip(reversed_array, fps=fps)
            
        except ImportError:
            # scipy not available, fall through to moviepy method
            pass
        except Exception as e:
            # If direct file read fails, fall through to moviepy method
            print(f"Warning: Could not read file directly ({e}), using moviepy method")
    
    # Fallback: use moviepy (or if file_path not provided)
    duration = clip.duration
    if duration is None or duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}")
    
    # Manually construct time array that's guaranteed to be within bounds
    n_samples = int(np.ceil(duration * fps))
    if n_samples <= 0:
        raise ValueError(f"No samples to process: duration={duration}, fps={fps}")
    
    # Generate time points from 0 to just before duration
    # Stay well within bounds to avoid edge issues
    epsilon = max(1.0 / fps, 1e-6)  # At least one sample period
    max_time = max(0, duration - epsilon)
    tt = np.linspace(0, max_time, n_samples)
    
    # Double-check bounds
    tt = np.clip(tt, 0, duration - 1e-6)
    
    audio_array = clip.get_frame(tt)
    
    if audio_array is None or len(audio_array) == 0:
        raise ValueError("Could not extract audio array from clip")
    
    reversed_array = np.flip(audio_array, axis=0).copy()
    
    from moviepy.audio.AudioClip import AudioArrayClip
    return AudioArrayClip(reversed_array, fps=fps)


from datafawn.soundscape.audio_utils import (
    frame_to_timestamp,
    create_sound_clip,
)
from datafawn.soundscape.event_utils import get_speed_from_zeni


def soundscape_auto(
    input_video_path: Union[str, Path],
    events_dict: Dict[str, Any],
    notes_folder: Union[str, Path] = "sounds/custom_tone",
    output_path: Optional[Union[str, Path]] = None,
    std_dev: float = 1.5,
    speed_threshold: Optional[float] = 0.8
) -> str:
    """
    Generate a soundscape by automatically selecting notes based on movement speed.
    
    For each event, randomly samples a note from a major scale (C, D, E, F, G, A, B).
    Front feet sample from C6 to G7, while back feet sample from C5 to G6 (one
    octave lower). The sampling is weighted by speed: higher relative speeds
    increase the probability of selecting higher notes using a normal distribution.
    
    Parameters
    -----------
    input_video_path : str or Path
        Path to input video file
    events_dict : Dict
        Dictionary of events from the pipeline.
        Structure: {(scorer, individual): {event_type: [frame_numbers]}}
        Example: {
            ('scorer1', 'animal0'): {
                'front_left_paw_strike': [10, 50, 90],
                'back_right_paw_strike': [30, 70, 110]
            }
        }
    notes_folder : str or Path, default="sounds/custom_tone"
        Path to folder containing note files (e.g., "C5.wav", "E5.wav", etc.)
    output_path : str or Path, optional
        Path for output video. If None, generates default path.
    std_dev : float, default=1.5
        Standard deviation for note sampling distribution. Higher values allow
        more randomness around the speed-based center note.
    speed_threshold : float, optional
        Speed threshold (0.0-1.0) for applying reverse effect. When speed
        crosses this threshold, the audio clip will be reversed. If None,
        no reverse effect is applied.
    
    Returns
    -------
    str
        Path to the output video file
    """
    input_video_path = Path(input_video_path)
    
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input Video file not found: {input_video_path}")
    
    # Load video
    print(f"Loading video: {input_video_path}")
    video = VideoFileClip(str(input_video_path))
    
    # Get FPS
    video_fps = video.fps
    print(f"Video FPS: {video_fps}")
    print(f"Video duration: {video.duration:.2f} seconds")
    
    # Prepare notes: filter to major scale (C, D, E, F, G, A, B)
    notes_folder = Path(notes_folder)
    if not notes_folder.exists():
        raise FileNotFoundError(f"Notes folder not found: {notes_folder}")
    
    # Define major scale notes for back feet (C5 to G6) and front feet (C6 to G7)
    # Major scale pattern: C, D, E, F, G, A, B repeated across octaves
    # Back feet: C5, D5, E5, F5, G5, A5, B5, C6, D6, E6, F6, G6
    # Front feet: C6, D6, E6, F6, G6, A6, B6, C7, D7, E7, F7, G7 (one octave higher)
    back_feet_notes = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6"]
    front_feet_notes = ["C6", "D6", "E6", "F6", "G6", "A6", "B6", "C7", "D7", "E7", "F7", "G7"]
    
    notes_folder_path = Path(notes_folder)
    
    # Load back feet notes
    back_feet_files = []
    for note_name in back_feet_notes:
        note_file = notes_folder_path / f"{note_name}.wav"
        if note_file.exists():
            back_feet_files.append(note_file)
        else:
            print(f"Warning: Note file not found: {note_file}, skipping...")
    
    # Load front feet notes
    front_feet_files = []
    for note_name in front_feet_notes:
        note_file = notes_folder_path / f"{note_name}.wav"
        if note_file.exists():
            front_feet_files.append(note_file)
        else:
            print(f"Warning: Note file not found: {note_file}, skipping...")
    
    if len(back_feet_files) == 0:
        raise ValueError(f"No valid back feet note files found in {notes_folder}")
    if len(front_feet_files) == 0:
        raise ValueError(f"No valid front feet note files found in {notes_folder}")
    
    print(f"\nFound {len(back_feet_files)} back feet note files: {[f.name for f in back_feet_files]}")
    print(f"Found {len(front_feet_files)} front feet note files: {[f.name for f in front_feet_files]}")
    
    # Calculate speed array using get_speed_from_zeni
    # Wrap events_dict in results dict format
    results_dict = {'events': events_dict}
    speed_array = get_speed_from_zeni(results_dict)
    print(f"Speed array calculated: {len(speed_array)} frames")
    
    # Create audio clips for all events from all individuals
    all_audio_clips = []
    
    # Iterate through all individuals
    for (scorer, individual), event_dict in events_dict.items():
        print(f"\n{'='*60}")
        print(f"Processing individual: {individual} (scorer: {scorer})")
        print(f"{'='*60}")
        
        # Iterate through each event type for this individual
        for event_name, strike_frames in event_dict.items():
            print(f"\n  Processing {event_name}:")
            if len(strike_frames) == 0:
                print(f"    No strikes found for {event_name}, skipping...")
                continue
            
            print(f"    Found {len(strike_frames)} strikes")
            
            # Convert frame numbers to timestamps
            strike_times = [frame_to_timestamp(frame, video_fps) for frame in strike_frames]
            print(f"    Strike timestamps: {[f'{t:.2f}s' for t in strike_times[:3]]}..." if len(strike_times) > 3 else f"    Strike timestamps: {[f'{t:.2f}s' for t in strike_times]}")
            
            # Determine if this is a front or back foot event
            is_front_foot = event_name.startswith('front_')
            note_files = front_feet_files if is_front_foot else back_feet_files
            
            # Create audio clips for each strike with speed-based note selection
            event_audio_clips = []
            note_selections = []  # Track selections for summary
            
            for frame, strike_time in zip(strike_frames, strike_times):
                # Get speed for this frame (handle out-of-bounds)
                if frame < len(speed_array):
                    speed = speed_array[frame]
                else:
                    speed = 0.0
                    if len(event_audio_clips) == 0:  # Only warn once per event type
                        print(f"    Warning: Some frames beyond speed array, using speed 0.0")
                
                # Map speed (0.0-1.0) to note index (0 to len(note_files)-1) for distribution center
                center_index = speed * (len(note_files) - 1)
                
                # Sample note index using normal distribution
                sampled_index = np.random.normal(center_index, std_dev)
                
                # Clamp to valid range and convert to integer
                sampled_index = int(np.clip(sampled_index, 0, len(note_files) - 1))
                
                # Select note file
                note_path = note_files[sampled_index]
                
                # Check if we should reverse the audio (speed crosses threshold)
                should_reverse = (speed_threshold is not None and speed >= speed_threshold)
                note_selections.append((speed, note_path.name, should_reverse))
                
                # Load note file and create audio clip
                original_clip = AudioFileClip(str(note_path))                
                
                # Step 1: Reverse first (if needed) - operates on full clip
                if should_reverse:
                    note_clip = audio_time_mirror(original_clip, file_path=note_path)
                else:
                    note_clip = original_clip
                
                # Step 3: Position the clip at strike_time
                note_clip = note_clip.with_start(strike_time)
                
                event_audio_clips.append(note_clip)
           
            # Print summary of note selections
            if len(note_selections) > 0:
                avg_speed = np.mean([s[0] for s in note_selections])
                note_counts = {}
                reversed_count = 0
                for speed, note, is_reversed in note_selections:
                    note_counts[note] = note_counts.get(note, 0) + 1
                    if is_reversed:
                        reversed_count += 1
                top_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                reverse_info = f", {reversed_count} reversed" if speed_threshold is not None else ""
                print(f"    Average speed: {avg_speed:.3f}, top notes: {top_notes}{reverse_info}")
            
            print(f"    Created {len(event_audio_clips)} audio clips for {event_name}")
            all_audio_clips.extend(event_audio_clips)
    
    print(f"\nTotal audio clips created: {len(all_audio_clips)}")
    
    # Combine all audio sources: original video audio and strike sounds
    audio_clips_to_composite = []
    
    # Add original video audio if present
    if video.audio is not None:
        original_audio = video.audio
        audio_clips_to_composite.append(original_audio)
        print("Original video audio included")
    else:
        print("No original video audio found, skipping...")
    
    # Add all strike sounds
    audio_clips_to_composite.extend(all_audio_clips)
    
    # Create final composite audio
    if len(audio_clips_to_composite) > 0:
        final_audio = CompositeAudioClip(audio_clips_to_composite)
    else:
        final_audio = None
    
    # Set the audio to the video
    if final_audio is not None:
        final_video = video.with_audio(final_audio)
    else:
        final_video = video
    
    if output_path is None:
        output_path = input_video_path.parent / f"{input_video_path.stem}_with_sounds.mp4"
    else:
        output_path = Path(output_path)
    
    # Write the final video
    print(f"\nWriting output video to: {output_path}")
    final_video.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        fps=video_fps
    )
    
    # Clean up
    final_video.close()
    video.close()
    for clip in all_audio_clips:
        clip.close()
    
    print(f"Done! Output saved to: {output_path}")
    return str(output_path)