"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

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
        notes_folder: Union[str, Path] = "sounds/custom_tone",
        std_dev: float = 1.5,
        speed_threshold: Optional[float] = 6.0,
        speed_window: int = 60,
        backing_track_path: Optional[Union[str, Path]] = None,
        backing_track_base_volume: float = 1.0,
        backing_track_max_volume: float = 1.0,
        backing_track_volume_curve: float = 3.0,
        show_speed_plot: bool = False,
        speed_plot_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the SoundScapeAuto generator.
        
        Parameters
        -----------
        notes_folder : str or Path, default="sounds/custom_tone"
            Path to folder containing note files (e.g., "C5.wav", "E5.wav", etc.)
        std_dev : float, default=1.5
            Standard deviation for note sampling distribution. Higher values allow
            more randomness around the speed-based center note.
        speed_threshold : float, default=6.0
            ABSOLUTE speed threshold (footfalls per window) for applying reverse
            effect and for backing track to reach max volume. Typical values: 4-10.
        speed_window : int, default=60
            Size of rolling window (in frames) for speed calculation. Larger values
            produce smoother speed curves, smaller values are more responsive.
        backing_track_path : str or Path, optional
            Path to backing track audio file. If provided, the backing track volume
            will be continuously scaled by the speed_array (louder when faster).
        backing_track_base_volume : float, default=0.5
            Minimum volume when speed is 0.0, relative to original sound level.
        backing_track_max_volume : float, default=1.0
            Maximum volume when speed reaches threshold, relative to original sound level.
        backing_track_volume_curve : float, default=3.0
            Power curve for volume scaling. Higher values keep volume near base
            longer, only ramping up close to speed_threshold.
            - 1.0 = linear, 2.0 = quadratic, 3.0 = cubic (default), 4.0+ = more extreme
        show_speed_plot : bool, default=False
            If True, displays a line graph of speed over time (useful for choosing threshold).
        speed_plot_path : str or Path, optional
            If provided, saves the speed plot to this path instead of displaying it.
        """
        self.notes_folder = notes_folder
        self.std_dev = std_dev
        self.speed_threshold = speed_threshold
        self.speed_window = speed_window
        self.backing_track_path = backing_track_path
        self.backing_track_base_volume = backing_track_base_volume
        self.backing_track_max_volume = backing_track_max_volume
        self.backing_track_volume_curve = backing_track_volume_curve
        self.show_speed_plot = show_speed_plot
        self.speed_plot_path = speed_plot_path
    
    
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
            output_path=output_path,
            notes_folder=self.notes_folder,
            std_dev=self.std_dev,
            speed_threshold=self.speed_threshold,
            speed_window=self.speed_window,
            backing_track_path=self.backing_track_path,
            backing_track_base_volume=self.backing_track_base_volume,
            backing_track_max_volume=self.backing_track_max_volume,
            backing_track_volume_curve=self.backing_track_volume_curve,
            show_speed_plot=self.show_speed_plot,
            speed_plot_path=self.speed_plot_path
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
