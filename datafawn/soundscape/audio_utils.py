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


def create_backing_track_with_speed_scaling(
    backing_track_path: Union[str, Path],
    video_duration: float,
    speed_array: np.ndarray,
    video_fps: float,
    base_volume: float = 0.5,
    max_volume: float = 1.0
):
    """
    Create a backing track audio clip with volume continuously scaled by speed_array.
    
    The volume of the backing track is scaled frame-by-frame based on the speed values
    in speed_array. Higher speed values result in higher volume. The volume is linearly
    interpolated from base_volume (when speed = 0.0) to max_volume (when speed = 1.0).
    
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
    print(f"Loading backing track from: {backing_track_path}")
    sample_rate, raw_audio = wavfile.read(str(backing_track_path))
    backing_duration = len(raw_audio) / sample_rate
    print(f"Raw audio from file: sample_rate={sample_rate}, shape={raw_audio.shape}, dtype={raw_audio.dtype}, duration={backing_duration:.2f}s")
    print(f"Video duration: {video_duration:.2f} seconds")
    
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
    print(f"Converted audio: shape={audio_array.shape}, min={np.min(audio_array):.4f}, max={np.max(audio_array):.4f}")
    
    # Calculate how many samples we need for video duration
    n_samples_needed = int(np.round(video_duration * backing_fps))
    
    # If backing track is shorter than video, loop it
    if len(audio_array) < n_samples_needed:
        num_loops = (n_samples_needed // len(audio_array)) + 1
        print(f"Looping audio {num_loops} times to match video duration")
        audio_array = np.tile(audio_array, (num_loops, 1))
    
    # Crop to exact video duration
    audio_array = audio_array[:n_samples_needed]
    print(f"Cropped audio to {len(audio_array)} samples ({video_duration:.2f}s)")
    print(f"Audio stats after crop: min={np.min(audio_array):.4f}, max={np.max(audio_array):.4f}, mean={np.mean(audio_array):.4f}")
    
    # Calculate volume scaling for each audio sample
    # Generate time points for each sample
    sample_times = np.linspace(0, video_duration, len(audio_array))
    frame_numbers = (sample_times * video_fps).astype(int)
    frame_numbers = np.clip(frame_numbers, 0, len(speed_array) - 1)
    
    # Get speed values and map to volume
    # Linear interpolation: speed 0.0 -> base_volume, speed 1.0 -> max_volume
    # These are multipliers (can be > 1.0 for amplification)
    speed_values = speed_array[frame_numbers]
    volume_multipliers = base_volume + speed_values * (max_volume - base_volume)
    # Ensure non-negative
    volume_multipliers = np.clip(volume_multipliers, 0.0, None)
    
    # Debug: print volume statistics BEFORE applying
    print(f"Volume multiplier stats: min={np.min(volume_multipliers):.3f}, max={np.max(volume_multipliers):.3f}, mean={np.mean(volume_multipliers):.3f}")
    print(f"Original audio stats: min={np.min(audio_array):.3f}, max={np.max(audio_array):.3f}, mean={np.mean(np.abs(audio_array)):.3f}")
    
    # Apply volume scaling to audio array
    if audio_array.shape[1] >= 1:
        volume_multipliers = volume_multipliers[:, np.newaxis]
    
    scaled_audio_array = audio_array * volume_multipliers
    
    # Debug: print volume statistics AFTER applying
    print(f"Scaled audio stats: min={np.min(scaled_audio_array):.3f}, max={np.max(scaled_audio_array):.3f}, mean={np.mean(np.abs(scaled_audio_array)):.3f}")
    
    # Check if audio is being completely muted
    if np.max(np.abs(scaled_audio_array)) < 0.001:
        print(f"WARNING: Scaled audio is extremely quiet (max={np.max(np.abs(scaled_audio_array)):.6f})!")
        print(f"This suggests volume scaling may be too low or speed_array values are all zero.")
    
    # Normalize audio to fit in [-1.0, 1.0] range for WAV file
    # This preserves relative volume differences but prevents clipping
    max_abs_value = np.max(np.abs(scaled_audio_array))
    if max_abs_value > 1.0:
        # Normalize to prevent clipping while preserving relative volumes
        print(f"Normalizing audio (max={max_abs_value:.3f}) to fit in [-1.0, 1.0] range")
        scaled_audio_array = scaled_audio_array / max_abs_value
    else:
        # No normalization needed, but ensure it's in valid range
        scaled_audio_array = np.clip(scaled_audio_array, -1.0, 1.0)
    
    # Convert to int16 for WAV file
    scaled_audio_int16 = (scaled_audio_array * 32767).astype(np.int16)
    
    # Debug: check if audio has any signal
    max_int16 = np.max(np.abs(scaled_audio_int16))
    print(f"Audio int16 stats: max={max_int16}, should be > 0 for audible audio")
    
    # Write to temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    wavfile.write(temp_path, int(backing_fps), scaled_audio_int16)
    print(f"Wrote scaled audio to temporary file: {temp_path}")
    
    # VERIFY: Read the temp file back with scipy to confirm what's actually in it
    verify_rate, verify_data = wavfile.read(temp_path)
    print(f"VERIFICATION - Read temp file back: rate={verify_rate}, shape={verify_data.shape}, dtype={verify_data.dtype}")
    print(f"VERIFICATION - Temp file stats: min={np.min(verify_data)}, max={np.max(verify_data)}, mean={np.mean(np.abs(verify_data)):.1f}")
    
    # Load the temporary file as AudioFileClip (which handles duration correctly)
    scaled_clip = AudioFileClip(temp_path)
    
    # Verify the loaded clip has audio
    test_audio = scaled_clip.get_frame([0.0, 0.1, 0.2])
    print(f"Loaded clip test: got {len(test_audio)} samples, max abs value={np.max(np.abs(test_audio)):.6f}")
    
    # Store temp_path on the clip so we can clean it up later if needed
    scaled_clip._temp_file_path = temp_path
    
    # Ensure the backing track starts at time 0
    # This is important for CompositeAudioClip to position it correctly
    scaled_clip = scaled_clip.with_start(0)
    
    print(f"Backing track volume scaled by speed_array:")
    print(f"  - Base volume (speed=0.0): {base_volume:.2f}x (relative to original)")
    print(f"  - Max volume (speed=1.0): {max_volume:.2f}x (relative to original)")
    print(f"  - Volume range: {base_volume:.2f}x to {max_volume:.2f}x")
    print(f"Final backing track: duration={scaled_clip.duration:.2f}s, start={getattr(scaled_clip, 'start', 'unknown')}")
    
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

