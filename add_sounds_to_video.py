"""
Script to add sounds to a video based on paw strike timestamps from the zeni algorithm.

This script takes:
- A video file (.mp4)
- A strikes dictionary from zeni_algorithm (frame indices where paws hit ground)
- A sound file (e.g., chime.wav)
- Which paw to track (e.g., 'front_right_paw')

And creates a new video with the sound playing at each strike timestamp.
"""

from pathlib import Path
import json
import sys

# Import moviepy - try different methods for compatibility with different versions
# First, check if moviepy is installed at all
try:
    import moviepy
    moviepy_version = getattr(moviepy, '__version__', 'unknown')
except ImportError:
    raise ImportError(
        "moviepy is not installed in your current Python environment.\n"
        f"Python path: {sys.executable}\n"
        "Please install it:\n"
        "  conda activate DEEPLABCUT\n"
        "  pip install moviepy\n"
        "or\n"
        "  conda install -c conda-forge moviepy"
    )

# Try importing the classes
try:
    # Try the standard editor import (works for most versions)
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
except (ImportError, ModuleNotFoundError, TypeError, AttributeError) as e:
    # If editor import fails, try direct imports (for moviepy 2.x or broken editor.py)
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.audio.AudioClip import CompositeAudioClip
    except (ImportError, ModuleNotFoundError, AttributeError) as e2:
        # If both fail, provide helpful error message
        raise ImportError(
            f"Could not import moviepy classes (moviepy {moviepy_version} detected).\n"
            f"Tried both 'moviepy.editor' and direct imports.\n"
            f"First error: {e}\n"
            f"Second error: {e2}\n\n"
            f"Python executable: {sys.executable}\n"
            f"MoviePy location: {moviepy.__file__ if hasattr(moviepy, '__file__') else 'unknown'}\n\n"
            f"Try restarting your Jupyter kernel after installing moviepy."
        ) from e2


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


def add_sounds_to_video(video_path, strikes_dict, sound_path, paw_name, 
                       output_path=None, fps=None):
    """
    Add sounds to a video at timestamps corresponding to paw strikes.
    
    Parameters:
    -----------
    video_path : str or Path
        Path to input video file
    strikes_dict : dict
        Dictionary with paw names as keys and lists of frame indices as values.
        Example: {'front_right_paw': [180, 205, 270, ...], ...}
    sound_path : str or Path
        Path to sound file to play at each strike
    paw_name : str
        Which paw to track (e.g., 'front_right_paw', 'back_left_paw', etc.)
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
    sound_path = Path(sound_path)
    
    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not sound_path.exists():
        raise FileNotFoundError(f"Sound file not found: {sound_path}")
    if paw_name not in strikes_dict:
        raise ValueError(f"Paw '{paw_name}' not found in strikes_dict. Available: {list(strikes_dict.keys())}")
    
    # Load video
    print(f"Loading video: {video_path}")
    video = VideoFileClip(str(video_path))
    
    # Get FPS (use provided or from video)
    video_fps = fps if fps is not None else video.fps
    print(f"Video FPS: {video_fps}")
    print(f"Video duration: {video.duration:.2f} seconds")
    
    # Get strike frames for the specified paw
    strike_frames = strikes_dict[paw_name]
    print(f"Found {len(strike_frames)} strikes for {paw_name}")
    
    if len(strike_frames) == 0:
        print("Warning: No strikes found for this paw. Output video will have no sounds.")
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_with_sounds.mp4"
        else:
            output_path = Path(output_path)
        video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
        video.close()
        return str(output_path)
    
    # Convert frame numbers to timestamps
    strike_times = [frame_to_timestamp(frame, video_fps) for frame in strike_frames]
    print(f"Strike timestamps: {[f'{t:.2f}s' for t in strike_times[:5]]}..." if len(strike_times) > 5 else f"Strike timestamps: {[f'{t:.2f}s' for t in strike_times]}")
    
    # Load sound file to get its duration
    sound_clip_template = AudioFileClip(str(sound_path))
    sound_duration = sound_clip_template.duration
    sound_clip_template.close()
    print(f"Sound duration: {sound_duration:.2f} seconds")
    
    # Create audio clips for each strike
    audio_clips = []
    for strike_time in strike_times:
        clip = create_sound_clip(sound_path, sound_duration, strike_time, video.duration)
        if clip is not None:
            audio_clips.append(clip)
    
    print(f"Created {len(audio_clips)} audio clips")
    
    # Combine all strike sounds with the original video audio
    if video.audio is not None:
        # Composite the original audio with all strike sounds
        final_audio = CompositeAudioClip([video.audio] + audio_clips)
    else:
        # No original audio, just use the strike sounds
        if len(audio_clips) > 0:
            final_audio = CompositeAudioClip(audio_clips)
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
        output_path = video_path.parent / f"{video_path.stem}_{paw_name}_with_sounds.mp4"
    else:
        output_path = Path(output_path)
    
    # Write the final video
    print(f"Writing output video to: {output_path}")
    final_video.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        fps=video_fps
    )
    
    # Clean up
    final_video.close()
    video.close()
    for clip in audio_clips:
        clip.close()
    
    print(f"Done! Output saved to: {output_path}")
    return str(output_path)


def add_sounds_for_all_paws(video_path, strikes_dict, paw_sound_map, 
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


# Example usage
if __name__ == "__main__":
    # Example: Add chime sounds when front_right_paw hits the ground
    video_path = "videos/deer2.mp4"
    sound_path = "chime.wav"  # You'll need to provide a sound file
    paw_name = "front_right_paw"
    
    # Example strikes dictionary (you would get this from zeni_algorithm)
    strikes = {
        'front_left_paw': [88, 132, 163, 204, 223, 254, 287, 304, 325, 344, 363, 385, 483, 511],
        'front_right_paw': [180, 205, 270, 298, 316, 335, 350, 366, 392, 485],
        'back_left_paw': [120, 160, 180, 224, 251, 277, 327, 346, 365, 386],
        'back_right_paw': [116, 143, 166, 192, 224, 253, 280, 306, 325, 344, 356, 376, 385, 495]
    }
    
    # Uncomment to run:
    # add_sounds_to_video(video_path, strikes, sound_path, paw_name)

