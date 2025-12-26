"""
Utility functions for soundscape generation.
"""

from pathlib import Path
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips



def frame_to_timestamp(frame_number, fps):
    """Convert frame number to timestamp in seconds."""
    return frame_number / fps


def create_backing_track(backing_track_path, video_duration, volume=1.0):
    """
    Create a backing track audio clip that matches the video duration.
    Loops if shorter, crops if longer.
    
    Parameters:
    -----------
    backing_track_path : str or Path
        Path to the backing track audio file
    video_duration : float
        Target duration in seconds (video duration)
    volume : float, optional
        Volume multiplier (1.0 = 100%, 0.5 = 50%, etc.). Default is 1.0.
    
    Returns:
    --------
    AudioFileClip
        Audio clip with duration matching video_duration and adjusted volume
    """
    backing_track_path = Path(backing_track_path)
    if not backing_track_path.exists():
        raise FileNotFoundError(f"Backing track file not found: {backing_track_path}")
    
    # Load the backing track
    backing_clip = AudioFileClip(str(backing_track_path))
    backing_duration = backing_clip.duration
    
    print(f"Backing track duration: {backing_duration:.2f} seconds")
    print(f"Video duration: {video_duration:.2f} seconds")
    
    # If backing track is shorter, loop it
    if backing_duration < video_duration:
        # Calculate how many times we need to loop
        num_loops = int(video_duration / backing_duration) + 1
        print(f"Backing track is shorter, looping {num_loops} times")
        
        # Create list of clips to concatenate
        clips_to_loop = [backing_clip] * num_loops
        
        # Concatenate to create looped version
        try:
            looped_clip = concatenate_audioclips(clips_to_loop)
        except Exception as e:
            # If concatenate fails, close the original clip and raise error
            backing_clip.close()
            raise RuntimeError(f"Could not loop backing track: {e}") from e
        
        # Close the original clip since we've created a new one
        backing_clip.close()
        backing_clip = looped_clip
    
    # If backing track is longer, crop it
    if backing_clip.duration > video_duration:
        print(f"Backing track is longer, cropping to {video_duration:.2f} seconds")
        original_clip = backing_clip
        backing_clip = backing_clip.with_end(video_duration)
        # Don't close original_clip here - subclipped clip may still need it
        # It will be garbage collected when backing_clip is no longer referenced
    
    # Apply volume adjustment if needed
    if volume != 1.0:
        backing_clip = backing_clip.with_volume_scaled(volume)
        print(f"Backing track volume set to {volume * 100:.0f}%")
    
    print(f"Final backing track duration: {backing_clip.duration:.2f} seconds")
    return backing_clip


def create_sound_clip(sound_path, duration, start_time, video_duration, volume=1.0):
    """
    Create an audio clip from a sound file, positioned at a specific time.
    Uses moviepy 2.2.1 API (subclip and with_start methods).
    
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
    volume : float, optional
        Volume multiplier (1.0 = 100%, 0.5 = 50%, etc.). Default is 1.0.
    
    Returns:
    --------
    AudioFileClip or None
        Audio clip positioned at start_time with adjusted volume, or None if start_time > video_duration
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
    
    # Trim sound to fit within video if needed
    if actual_duration < sound_clip.duration:
        original_clip = sound_clip
        sound_clip = sound_clip.with_end(actual_duration)
        # Don't close original_clip here - subclipped clip may still need it
        # It will be garbage collected when sound_clip is no longer referenced
    
    # Position the clip at start_time
    sound_clip = sound_clip.with_start(start_time)
    
    # Apply volume adjustment if needed
    if volume != 1.0:
        sound_clip = sound_clip.with_volume_scaled(volume)
    
    return sound_clip

