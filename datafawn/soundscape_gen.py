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
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip

from datafawn.pipeline import SoundScapeGenerator
import numpy as np
import math

def frame_to_timestamp(frame_number, fps):
    """Convert frame number to timestamp in seconds."""
    return frame_number / fps



def get_speed_from_zeni(results : dict = {}, window : int = 30):
    '''
    Label frames in video based on how fast the animal is running at that time

    Parameters
    ------------

    results : dict
        Output from zeni algorithm

    window : int 
        Number of frames for window

    '''
    if 'events' not in results:
        raise ValueError("Please enter in a valid results_ dictionary")
    

    foot_falls = []
    for (scorer, individual), event_dict in results['events'].items():
        for foot, frames in event_dict.items():
            for frame in frames : 
                foot_falls.append(frame)
    
    foot_falls.sort()
    all_frames = np.zeros(shape=(foot_falls[-1] + 1), dtype=float)

    num_windows = math.ceil(len(all_frames) / window)

    highest_foot_falls_per_window = 0
    for i in range (num_windows):
        count = sum((i * window) <= x <= (i * window + window) for x in foot_falls)
        if count > highest_foot_falls_per_window:
            highest_foot_falls_per_window = count

    for i in range (num_windows):
        count = sum((i * window) <= x <= (i * window + window) for x in foot_falls)
        for f in foot_falls:
            if (i * window) <= f and f <= (i * window + window):
                all_frames[f] = count/highest_foot_falls_per_window
    
    # all_frames[frame] = 0 if there is no foot fall at that frame
    # all_frames[frame] != 0 if there is a foot fall at that frame
    # should be a value between 0 and 1
    # 1 meaning that the animal is moving at it's fastest
    # 0 meaning the animal is moving at it's slowest
    return all_frames




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


class SoundScapeFromConfig(SoundScapeGenerator):
    """
    """

    def __init__(
        self, 
        soundscape_config: Dict[str, Any],
    ):
        self.soundscape_config = soundscape_config

    
    def generate(
        self, 
        input_video_path: Union[str, Path], 
        events_dict: Dict[str, Any], 
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Add sounds for detected events to a video.
        
        Iterates through all individuals and adds sounds for each event type
        based on the event_sound_map in the config.
        
        Parameters:
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
        output_path : str or Path, optional
            Path for output video. If None, generates default path.
        
        Returns:
        --------
        str
            Path to the output video file
        """
        event_sound_map = self.soundscape_config['event_sound_map']

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
        
        # Create audio clips for all events from all individuals
        all_audio_clips = []
        
        # Iterate through all individuals
        for (scorer, individual), event_dict in events_dict.items():
            print(f"\n{'='*60}")
            print(f"Processing individual: {individual} (scorer: {scorer})")
            print(f"{'='*60}")
            
            # Iterate through each event type for this individual
            for event_name, strike_frames in event_dict.items():
                # Check if this event type has a sound mapped
                if event_name not in event_sound_map:
                    print(f"  Skipping {event_name}: no sound file mapped")
                    continue
                
                sound_path = Path(event_sound_map[event_name])
                if not sound_path.exists():
                    raise FileNotFoundError(f"Sound file not found for {event_name}: {sound_path}")
                
                print(f"\n  Processing {event_name}:")
                if len(strike_frames) == 0:
                    print(f"    No strikes found for {event_name}, skipping...")
                    continue

                print(f"    Found {len(strike_frames)} strikes")
                
                # Convert frame numbers to timestamps
                strike_times = [frame_to_timestamp(frame, video_fps) for frame in strike_frames]
                print(f"    Strike timestamps: {[f'{t:.2f}s' for t in strike_times[:3]]}..." if len(strike_times) > 3 else f"    Strike timestamps: {[f'{t:.2f}s' for t in strike_times]}")
                
                # Load sound file to get its duration
                sound_clip_template = AudioFileClip(str(sound_path))
                sound_duration = sound_clip_template.duration
                sound_clip_template.close()
                print(f"    Sound duration: {sound_duration:.2f} seconds")
                
                # Create audio clips for each strike
                event_audio_clips = []
                for strike_time in strike_times:
                    clip = create_sound_clip(sound_path, sound_duration, strike_time, video.duration)
                    if clip is not None:
                        event_audio_clips.append(clip)
                
                print(f"    Created {len(event_audio_clips)} audio clips for {event_name}")
                all_audio_clips.extend(event_audio_clips)
        
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