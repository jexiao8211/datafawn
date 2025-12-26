"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip

from datafawn.pipeline import SoundScapeGenerator
from datafawn.soundscape import (
    frame_to_timestamp,
    create_backing_track,
    create_sound_clip,
)


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
        backing_track = self.soundscape_config['backing_track']
        # Get volume settings from config, with defaults
        volume_config = self.soundscape_config.get('volume', {})
        backing_volume = volume_config.get('backing_track', 1.0)
        event_sounds_volume = volume_config.get('event_sounds', 1.0)
        original_video_volume = volume_config.get('original_video', 1.0)

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
                    clip = create_sound_clip(sound_path, sound_duration, strike_time, video.duration, volume=event_sounds_volume)
                    if clip is not None:
                        event_audio_clips.append(clip)
                
                print(f"    Created {len(event_audio_clips)} audio clips for {event_name}")
                all_audio_clips.extend(event_audio_clips)
        
        print(f"\nTotal audio clips created: {len(all_audio_clips)}")
        
        # Prepare backing track if provided
        backing_track_clip = None
        if backing_track:
            print(f"\n{'='*60}")
            print("Processing backing track")
            print(f"{'='*60}")
            backing_track_clip = create_backing_track(backing_track, video.duration, volume=backing_volume)
        
        # Combine all audio sources: backing track, original video audio, and strike sounds
        audio_clips_to_composite = []
        
        # Add backing track first (lowest layer)
        if backing_track_clip is not None:
            audio_clips_to_composite.append(backing_track_clip)
        
        # Add original video audio if present
        if video.audio is not None:
            original_audio = video.audio
            # Apply volume adjustment if needed
            if original_video_volume != 1.0:
                original_audio = original_audio.with_volume_scaled(original_video_volume)
                print(f"Original video audio volume set to {original_video_volume * 100:.0f}%")
            audio_clips_to_composite.append(original_audio)
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
        if backing_track_clip is not None:
            backing_track_clip.close()
        
        print(f"Done! Output saved to: {output_path}")
        return str(output_path)