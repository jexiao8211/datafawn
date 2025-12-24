"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from typing import Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import json
import sys
import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

from datafawn.pipeline import SoundScapeGenerator


def frame_to_timestamp(frame_number, fps):
    """Convert frame number to timestamp in seconds."""
    return frame_number / fps

def create_sound_clip(sound_path, duration, start_time, video_duration):
    """
    Create an audio clip from a sound file, positioned at a specific time.
    Compatible with moviepy 2.x API (uses with_* methods).
    
    Parameters:
    -----------
    sound_path : str or Path
        Path to the sound file (.wav, .mp3, etc.)
    duration : float
        Duration of the sound in seconds
    start_time : float
        When to start playing the sound (in seconds)
    video_duration : float
        Total duration of the video (to ensure sound doesn't exceed video)
    
    Returns:
    --------
    AudioFileClip or None
        Audio clip positioned at start_time, or None if start_time > video_duration
    """
    if start_time >= video_duration:
        return None
    
    # Load the sound file
    sound_clip = AudioFileClip(str(sound_path))
    
    # Trim sound if it would exceed video duration
    end_time = min(start_time + duration, video_duration)
    actual_duration = end_time - start_time
    
    if actual_duration <= 0:
        sound_clip.close()
        return None
    
    # Trim sound to fit within video
    # In moviepy 2.x, use with_subclip() or subclip() if available
    if actual_duration < sound_clip.duration:
        try:
            # Try with_subclip first (moviepy 2.x style)
            if hasattr(sound_clip, 'with_subclip'):
                sound_clip = sound_clip.with_subclip(0, actual_duration)
            # Fallback to subclip (moviepy 1.x, might still work in 2.x)
            elif hasattr(sound_clip, 'subclip'):
                sound_clip = sound_clip.subclip(0, actual_duration)
            # If neither works, try setting duration directly
            elif hasattr(sound_clip, 'with_duration'):
                sound_clip = sound_clip.with_duration(actual_duration)
            else:
                # If no trimming method available, use full clip
                # CompositeAudioClip will handle timing
                pass
        except (AttributeError, TypeError, ValueError) as e:
            # If trimming fails, use full clip - CompositeAudioClip will handle it
            pass
    
    # Position the clip at start_time - moviepy 2.x uses with_start()
    try:
        if hasattr(sound_clip, 'with_start'):
            # moviepy 2.x
            sound_clip = sound_clip.with_start(start_time)
        elif hasattr(sound_clip, 'set_start'):
            # moviepy 1.x
            sound_clip = sound_clip.set_start(start_time)
        else:
            # Fallback: try setting start attribute directly
            sound_clip.start = start_time
    except (AttributeError, TypeError) as e:
        # If positioning fails, try setting attribute directly
        try:
            sound_clip.start = start_time
        except:
            raise RuntimeError(f"Could not position audio clip at {start_time}s. MoviePy version may be incompatible.") from e
    
    return sound_clip


class SoundScapeGenerator(SoundScapeGenerator):
    """
    """

    def __init__(
        self, 

    ):
        pass

    
def generate(self, video_path, strikes_dict, paw_sound_map, 
                            output_path=None, fps=None):
    """
    Add different sounds for each paw to a video simultaneously.
    
    This function allows you to add different sounds for each paw in a single video,
    so you don't need to create separate videos for each paw.
    
    Parameters:
    -----------
    video_path : str or Path
        Path to input video file
    strikes_dict : dict
        Dictionary with paw names as keys and lists of frame indices as values.
        Example: {'front_right_paw': [180, 205, 270, ...], ...}
    paw_sound_map : dict
        Dictionary mapping paw names to sound file paths.
        Example: {
            'front_left_paw': 'sound1.wav',
            'front_right_paw': 'sound2.wav',
            'back_left_paw': 'sound3.wav',
            'back_right_paw': 'sound4.wav'
        }
        You can omit paws you don't want sounds for.
    output_path : str or Path, optional
        Path for output video. If None, auto-generates name.
    fps : float, optional
        Video FPS. If None, reads from video file.
    
    Returns:
    --------
    str
        Path to the output video file
    """
    video_path = Path(video_path)
    
    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Validate paw_sound_map
    for paw_name, sound_path in paw_sound_map.items():
        sound_path = Path(sound_path)
        if not sound_path.exists():
            raise FileNotFoundError(f"Sound file not found for {paw_name}: {sound_path}")
        if paw_name not in strikes_dict:
            print(f"Warning: Paw '{paw_name}' not found in strikes_dict. Skipping.")
    
    # Load video
    print(f"Loading video: {video_path}")
    video = VideoFileClip(str(video_path))
    
    # Get FPS (use provided or from video)
    video_fps = fps if fps is not None else video.fps
    print(f"Video FPS: {video_fps}")
    print(f"Video duration: {video.duration:.2f} seconds")
    
    # Create audio clips for all paws
    all_audio_clips = []
    
    for paw_name, sound_path in paw_sound_map.items():
        if paw_name not in strikes_dict:
            continue
            
        strike_frames = strikes_dict[paw_name]
        sound_path = Path(sound_path)
        
        print(f"\nProcessing {paw_name}:")
        print(f"  Found {len(strike_frames)} strikes")
        
        if len(strike_frames) == 0:
            print(f"  No strikes found for {paw_name}, skipping...")
            continue
        
        # Convert frame numbers to timestamps
        strike_times = [frame_to_timestamp(frame, video_fps) for frame in strike_frames]
        print(f"  Strike timestamps: {[f'{t:.2f}s' for t in strike_times[:3]]}..." if len(strike_times) > 3 else f"  Strike timestamps: {[f'{t:.2f}s' for t in strike_times]}")
        
        # Load sound file to get its duration
        sound_clip_template = AudioFileClip(str(sound_path))
        sound_duration = sound_clip_template.duration
        sound_clip_template.close()
        print(f"  Sound duration: {sound_duration:.2f} seconds")
        
        # Create audio clips for each strike of this paw
        paw_audio_clips = []
        for strike_time in strike_times:
            clip = create_sound_clip(sound_path, sound_duration, strike_time, video.duration)
            if clip is not None:
                paw_audio_clips.append(clip)
        
        print(f"  Created {len(paw_audio_clips)} audio clips for {paw_name}")
        all_audio_clips.extend(paw_audio_clips)
    
    print(f"\nTotal audio clips created: {len(all_audio_clips)}")
    
    # Combine all strike sounds with the original video audio
    if video.audio is not None:
        # Composite the original audio with all strike sounds
        final_audio = CompositeAudioClip([video.audio] + all_audio_clips)
    else:
        # No original audio, just use the strike sounds
        if len(all_audio_clips) > 0:
            final_audio = CompositeAudioClip(all_audio_clips)
        else:
            final_audio = None
    
    # Set the audio to the video
    # MoviePy 2.x uses with_audio, older versions use set_audio
    if final_audio is not None:
        if hasattr(video, 'with_audio'):
            final_video = video.with_audio(final_audio)
        elif hasattr(video, 'set_audio'):
            final_video = video.set_audio(final_audio)
        else:
            raise AttributeError(
                f"VideoFileClip has neither 'with_audio' nor 'set_audio' method. "
                f"MoviePy version may be incompatible."
            )
    else:
        final_video = video
    
    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_all_paws_with_sounds.mp4"
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


def load_strikes_from_json(json_path):
    """Load strikes dictionary from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_strikes_to_json(strikes_dict, json_path):
    """Save strikes dictionary to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(strikes_dict, f, indent=2)