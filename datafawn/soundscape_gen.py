"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from ast import Pass
from typing import Dict, Any, Optional, Union
from pathlib import Path

from datafawn.pipeline import SoundScapeGenerator
from datafawn.soundscape.soundscape_from_config import generate_soundscape_from_config
from datafawn.soundscape.soundscape_auto import soundscape_auto

class SoundScapeAuto(SoundScapeGenerator):
    """
    Soundscape generator that creates audio overlays from event data based on configuration.
    
    Wraps the generate_soundscape_from_config function.
    """

    def __init__(
        self, 
    ):
        pass
    
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
        output_path : str or Path, optional
            Path for output video. If None, generates default path.
        
        Returns
        -------
        str
            Path to the output video file
        """
        return soundscape_auto(
            input_video_path=input_video_path,
            events_dict=events_dict,
            output_path=output_path
        )


class SoundScapeFromConfig(SoundScapeGenerator):
    """
    Soundscape generator that creates audio overlays from event data based on configuration.
    
    Wraps the generate_soundscape_from_config function.
    """

    def __init__(
        self, 
        soundscape_config: Dict[str, Any],
    ):
        """
        Initialize the SoundScapeFromConfig generator.
        
        Parameters
        -----------
        soundscape_config : Dict[str, Any]
            Configuration dictionary containing:
            - 'event_sound_map': Dict mapping event names to sound file paths
            - 'backing_track': Optional path to backing track audio file
            - 'volume': Optional dict with 'backing_track', 'event_sounds', 'original_video' keys
        """
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
        output_path : str or Path, optional
            Path for output video. If None, generates default path.
        
        Returns
        -------
        str
            Path to the output video file
        """
        return generate_soundscape_from_config(
            input_video_path=input_video_path,
            events_dict=events_dict,
            soundscape_config=self.soundscape_config,
            output_path=output_path
        )
