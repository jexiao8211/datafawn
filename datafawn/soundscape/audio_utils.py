"""
Utility functions for soundscape generation.
"""

from pathlib import Path
from typing import Union
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips
import numpy as np


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
    
    print(f"Loading backing track: {backing_duration:.2f}s")
    
    # If backing track is shorter, loop it
    if backing_duration < video_duration:
        num_loops = int(video_duration / backing_duration) + 1
        print(f"  Looping {num_loops}x to match video")
        clips_to_loop = [backing_clip] * num_loops
        try:
            looped_clip = concatenate_audioclips(clips_to_loop)
        except Exception as e:
            backing_clip.close()
            raise RuntimeError(f"Could not loop backing track: {e}") from e
        backing_clip.close()
        backing_clip = looped_clip
    
    # If backing track is longer, crop it
    if backing_clip.duration > video_duration:
        print(f"  Cropping to {video_duration:.2f}s")
        backing_clip = backing_clip.with_end(video_duration)
    
    # Apply volume adjustment if needed
    if volume != 1.0:
        backing_clip = backing_clip.with_volume_scaled(volume)
        print(f"  Volume: {volume * 100:.0f}%")
    
    return backing_clip


def create_backing_track_with_speed_scaling(
    backing_track_path: Union[str, Path],
    video_duration: float,
    speed_array: np.ndarray,
    video_fps: float,
    base_volume: float = 0.5,
    max_volume: float = 1.0,
    speed_threshold: float = 0.8,
    volume_curve: float = 3.0
):
    """
    Create a backing track audio clip with volume continuously scaled by speed_array.
    
    The volume of the backing track is scaled frame-by-frame based on the speed values
    in speed_array. Higher speed values result in higher volume. Volume scaling uses
    a power curve that keeps volume near base_volume until speed approaches the
    threshold, then ramps up quickly to max_volume.
    
    Parameters:
    -----------
    backing_track_path : str or Path
        Path to the backing track audio file
    video_duration : float
        Target duration in seconds (video duration)
    speed_array : np.ndarray
        Array of speed values (0.0-1.0) for each frame, where index corresponds to frame number
    video_fps : float
        Frames per second of the video (used to map time to frame numbers)
    base_volume : float, optional
        Minimum volume when speed is 0.0, relative to original sound level.
        Default is 0.5 (50% of original volume).
    max_volume : float, optional
        Maximum volume when speed is 1.0, relative to original sound level.
        Default is 1.0 (100% of original volume).
    speed_threshold : float, optional
        Speed value at which volume reaches max_volume. Speeds above this are clamped.
        Default is 0.8.
    volume_curve : float, optional
        Power curve exponent for volume scaling. Higher values keep volume near
        base_volume longer, only ramping up close to threshold.
        - 1.0 = linear scaling
        - 2.0 = quadratic (moderate curve)
        - 3.0 = cubic (default, volume stays low until near threshold)
        - 4.0+ = more extreme, volume stays very flat until very close to threshold
    
    Returns:
    --------
    AudioFileClip
        Audio clip with duration matching video_duration and volume scaled by speed_array
    """
    backing_track_path = Path(backing_track_path)
    if not backing_track_path.exists():
        raise FileNotFoundError(f"Backing track file not found: {backing_track_path}")
    
    import tempfile
    from scipy.io import wavfile
    
    # Read the backing track directly with scipy for accurate audio data
    # (MoviePy's get_frame() doesn't return proper audio waveforms)
    print(f"Loading backing track: {backing_track_path}")
    sample_rate, raw_audio = wavfile.read(str(backing_track_path))
    backing_duration = len(raw_audio) / sample_rate
    print(f"  Duration: {backing_duration:.2f}s, Sample rate: {sample_rate}Hz")
    
    # Convert to float32 normalized to [-1.0, 1.0]
    if raw_audio.dtype == np.int16:
        audio_array = raw_audio.astype(np.float32) / 32768.0
    elif raw_audio.dtype == np.int32:
        audio_array = raw_audio.astype(np.float32) / 2147483648.0
    elif raw_audio.dtype == np.float32:
        audio_array = raw_audio
    else:
        audio_array = raw_audio.astype(np.float32)
    
    # Ensure 2D (samples, channels)
    if len(audio_array.shape) == 1:
        audio_array = audio_array[:, np.newaxis]
    
    backing_fps = sample_rate
    
    # Calculate how many samples we need for video duration
    n_samples_needed = int(np.round(video_duration * backing_fps))
    
    # If backing track is shorter than video, loop it
    if len(audio_array) < n_samples_needed:
        num_loops = (n_samples_needed // len(audio_array)) + 1
        print(f"  Looping {num_loops}x to match video duration")
        audio_array = np.tile(audio_array, (num_loops, 1))
    elif backing_duration > video_duration:
        print(f"  Cropping to {video_duration:.2f}s")
    
    # Crop to exact video duration
    audio_array = audio_array[:n_samples_needed]
    
    # Calculate volume scaling for each audio sample
    # Generate time points for each sample
    sample_times = np.linspace(0, video_duration, len(audio_array))
    frame_numbers = (sample_times * video_fps).astype(int)
    frame_numbers = np.clip(frame_numbers, 0, len(speed_array) - 1)
    
    # Map speed values to volume multipliers using power curve
    # This keeps volume near base_volume until speed approaches threshold
    speed_values = speed_array[frame_numbers]
    
    # Normalize speed relative to threshold (clamp at 1.0)
    normalized_speed = np.clip(speed_values / speed_threshold, 0.0, 1.0)
    
    # Apply power curve: values stay low until approaching 1.0, then ramp up
    # curve=1.0 is linear, curve=3.0 keeps volume flat until ~70% of threshold
    curved_speed = np.power(normalized_speed, volume_curve)
    
    # Map curved values to volume range
    volume_multipliers = base_volume + curved_speed * (max_volume - base_volume)
    volume_multipliers = np.clip(volume_multipliers, 0.0, None)  # Ensure non-negative
    
    print(f"  Volume scaling: curve={volume_curve}, threshold={speed_threshold}")
    print(f"  Volume range: {np.min(volume_multipliers):.2f}x - {np.max(volume_multipliers):.2f}x (mean: {np.mean(volume_multipliers):.2f}x)")
    
    # Apply volume scaling to audio array
    if audio_array.shape[1] >= 1:
        volume_multipliers = volume_multipliers[:, np.newaxis]
    
    scaled_audio_array = audio_array * volume_multipliers
    
    # Warn if audio is extremely quiet
    if np.max(np.abs(scaled_audio_array)) < 0.001:
        print(f"  WARNING: Audio is extremely quiet - check volume settings or speed_array")
    
    # Normalize audio to fit in [-1.0, 1.0] range for WAV file
    max_abs_value = np.max(np.abs(scaled_audio_array))
    if max_abs_value > 1.0:
        scaled_audio_array = scaled_audio_array / max_abs_value
    else:
        scaled_audio_array = np.clip(scaled_audio_array, -1.0, 1.0)
    
    # Convert to int16 and write to temporary WAV file
    scaled_audio_int16 = (scaled_audio_array * 32767).astype(np.int16)
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    wavfile.write(temp_path, int(backing_fps), scaled_audio_int16)
    
    # Load as AudioFileClip for proper duration handling
    scaled_clip = AudioFileClip(temp_path)
    scaled_clip._temp_file_path = temp_path  # Store for cleanup
    scaled_clip = scaled_clip.with_start(0)  # Ensure starts at time 0
    
    print(f"  Speed-scaled backing track ready: {scaled_clip.duration:.2f}s")
    
    return scaled_clip


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

