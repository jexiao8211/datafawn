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

from datafawn.soundscape.audio_utils import (
    frame_to_timestamp,
    create_sound_clip,
    audio_time_mirror,
    create_backing_track_with_speed_scaling,
)
from datafawn.soundscape.event_utils import (
    get_speed_absolute,
    print_speed_stats,
    plot_speed_array,
)


def soundscape_auto(
    input_video_path: Union[str, Path],
    events_dict: Dict[str, Any],
    notes_folder: Union[str, Path] = "sounds/custom_tone",
    output_path: Optional[Union[str, Path]] = None,
    std_dev: float = 1.5,
    speed_threshold: Optional[float] = 6.0,
    speed_window: int = 60,
    backing_track_path: Optional[Union[str, Path]] = 'sounds/calm_ambient_backing.wav',
    backing_track_base_volume: float = 0.5,
    backing_track_max_volume: float = 1.0,
    backing_track_volume_curve: float = 3.0,
    show_speed_plot: bool = False,
    speed_plot_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a soundscape by automatically selecting notes based on movement speed.
    
    For each event, randomly samples a note from a major scale (C, D, E, F, G, A, B).
    Front feet sample from C6 to G7, while back feet sample from C5 to G6 (one
    octave lower). The sampling is weighted by speed: higher relative speeds
    increase the probability of selecting higher notes using a normal distribution.
    
    Speed is measured in ABSOLUTE terms (footfalls per window), not relative to
    the video's maximum. This allows consistent thresholds across different videos.
    
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
    speed_threshold : float, default=6.0
        ABSOLUTE speed threshold (footfalls per window) for applying reverse effect
        and for backing track to reach max volume. When speed crosses this threshold,
        the audio clip will be reversed. Set to None to disable reverse effect.
        Typical values: 4-10 depending on animal and video.
    speed_window : int, default=60
        Size of rolling window (in frames) for speed calculation. Larger values
        produce smoother speed curves, smaller values are more responsive.
        At 30 fps, 60 frames = 2 seconds of smoothing.
    backing_track_path : str or Path, optional
        Path to backing track audio file. If provided, the backing track volume
        will be continuously scaled by the speed_array (louder when faster).
    backing_track_base_volume : float, default=0.5
        Minimum volume when speed is 0.0, relative to original sound level.
    backing_track_max_volume : float, default=1.0
        Maximum volume when speed reaches threshold, relative to original sound level.
    backing_track_volume_curve : float, default=3.0
        Power curve for volume scaling. Higher values keep volume near base_volume
        longer, only ramping up close to speed_threshold.
        - 1.0 = linear scaling
        - 2.0 = quadratic (moderate curve)
        - 3.0 = cubic (default, stays flat until near threshold)
        - 4.0+ = more extreme curve
    show_speed_plot : bool, default=False
        If True, displays a line graph of speed over time (useful for choosing threshold).
    speed_plot_path : str or Path, optional
        If provided, saves the speed plot to this path instead of displaying it.
    
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
    back_feet_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6", "B6"]
    front_feet_notes = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6", "B6", "C7", "D7", "E7", "F7", "G7", "A7", "B7"]
    
    notes_folder_path = Path(notes_folder)
    
    # Load back feet notes
    back_feet_files = []
    for note_name in back_feet_notes:
        note_file = notes_folder_path / f"{note_name}.wav"
        if note_file.exists():
            back_feet_files.append(note_file)
    
    # Load front feet notes
    front_feet_files = []
    for note_name in front_feet_notes:
        note_file = notes_folder_path / f"{note_name}.wav"
        if note_file.exists():
            front_feet_files.append(note_file)
    
    if len(back_feet_files) == 0:
        raise ValueError(f"No valid back feet note files found in {notes_folder}")
    if len(front_feet_files) == 0:
        raise ValueError(f"No valid front feet note files found in {notes_folder}")
    
    print(f"Loaded {len(back_feet_files)} back feet notes, {len(front_feet_files)} front feet notes")
    
    # Calculate ABSOLUTE speed array (footfalls per window, not normalized)
    results_dict = {'events': events_dict}
    speed_array = get_speed_absolute(results_dict, window=speed_window)
    
    # Print speed statistics to help choose thresholds
    print_speed_stats(speed_array, fps=video_fps)
    
    # Show/save speed plot if requested
    if show_speed_plot or speed_plot_path:
        plot_speed_array(
            speed_array, 
            fps=video_fps, 
            title=f"Speed Over Time - {input_video_path.name}",
            save_path=str(speed_plot_path) if speed_plot_path else None,
            threshold=speed_threshold
        )
    
    # Create audio clips for all events from all individuals
    all_audio_clips = []
    
    # Iterate through all individuals
    for (scorer, individual), event_dict in events_dict.items():
        print(f"\n{'='*60}")
        print(f"Processing individual: {individual} (scorer: {scorer})")
        print(f"{'='*60}")
        
        # Iterate through each event type for this individual
        for event_name, strike_frames in event_dict.items():
            if len(strike_frames) == 0:
                continue
            
            # Convert frame numbers to timestamps
            strike_times = [frame_to_timestamp(frame, video_fps) for frame in strike_frames]
            
            # Determine if this is a front or back foot event
            is_front_foot = event_name.startswith('front_')
            note_files = front_feet_files if is_front_foot else back_feet_files
            
            # Create audio clips for each strike with speed-based note selection
            event_audio_clips = []
            note_selections = []  # Track selections for summary
            
            for frame, strike_time in zip(strike_frames, strike_times):
                # Get absolute speed for this frame (footfalls per window)
                if frame < len(speed_array):
                    speed = speed_array[frame]
                else:
                    speed = 0.0
                    if len(event_audio_clips) == 0:  # Only warn once per event type
                        print(f"    Warning: Some frames beyond speed array, using speed 0.0")
                
                # Normalize speed to 0-1 for note selection (relative to threshold)
                # Speed at or above threshold maps to 1.0 (highest notes)
                effective_threshold = speed_threshold if speed_threshold is not None else max(np.max(speed_array), 1.0)
                speed_normalized = min(speed / effective_threshold, 1.0)
                
                # Map normalized speed to note index for distribution center
                # Leave room for distribution tails (std_dev away from edges)
                # This ensures both left and right tails have room to sample from
                min_center = std_dev
                max_center = len(note_files) - 1 - std_dev
                center_index = min_center + speed_normalized * (max_center - min_center)
                
                # Sample note index using normal distribution
                sampled_index = np.random.normal(center_index, std_dev)
                
                # Clamp to valid range and convert to integer
                sampled_index = int(np.clip(sampled_index, 0, len(note_files) - 1))
                
                # Select note file
                note_path = note_files[sampled_index]
                
                # Check if we should reverse the audio (absolute speed crosses threshold)
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
           
            # Summary for this event
            if len(note_selections) > 0:
                reversed_count = sum(1 for _, _, r in note_selections if r)
                reverse_info = f", {reversed_count} reversed" if speed_threshold is not None and reversed_count > 0 else ""
                print(f"    {event_name}: {len(event_audio_clips)} clips{reverse_info}")
            
            all_audio_clips.extend(event_audio_clips)
    
    print(f"\nTotal audio clips created: {len(all_audio_clips)}")
    
    # Prepare backing track with speed scaling if provided
    backing_track_clip = None
    if backing_track_path is not None:
        backing_track_path_obj = Path(backing_track_path)
        if backing_track_path_obj.exists():
            # Use speed_threshold for backing track (default to max speed if None)
            bt_threshold = speed_threshold if speed_threshold is not None else max(np.max(speed_array), 1.0)
            print(f"\nProcessing backing track (volume: {backing_track_base_volume:.1f}x - {backing_track_max_volume:.1f}x at speed>={bt_threshold:.1f})")
            backing_track_clip = create_backing_track_with_speed_scaling(
                backing_track_path=backing_track_path,
                video_duration=video.duration,
                speed_array=speed_array,
                video_fps=video_fps,
                base_volume=backing_track_base_volume,
                max_volume=backing_track_max_volume,
                speed_threshold=bt_threshold,
                volume_curve=backing_track_volume_curve
            )
        else:
            print(f"Warning: Backing track not found: {backing_track_path}")
    
    # Combine all audio sources: backing track, original video audio, and strike sounds
    audio_clips_to_composite = []
    
    # Add backing track first (lowest layer)
    if backing_track_clip is not None:
        audio_clips_to_composite.append(backing_track_clip)
        print(f"Adding backing track: {backing_track_clip.duration:.2f}s")
    
    # Add original video audio if present
    if video.audio is not None:
        audio_clips_to_composite.append(video.audio)
        print(f"Adding original video audio: {video.audio.duration:.2f}s")
    
    # Add all strike sounds
    audio_clips_to_composite.extend(all_audio_clips)
    
    # Create final composite audio
    print(f"\nCompositing {len(audio_clips_to_composite)} audio clips...")
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
    if backing_track_clip is not None:
        backing_track_clip.close()
    
    print(f"Done! Output saved to: {output_path}")
    return str(output_path)