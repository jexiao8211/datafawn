"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from datafawn.pipeline import EventExtractor
from event_extraction.zeni import zeni_algorithm


class ZeniExtractor(EventExtractor):
    """
    Event extractor using the Zeni algorithm for foot strike detection.
    
    Wraps the zeni_algorithm function.
    """
    
    def __init__(
        self,
        # window_size: int = 5,
        # min_contact_duration: int = 3,
        # velocity_threshold: float = 10,
        # prominence_percentage: float = 0.05,
        # show_plots: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize the Zeni extractor.
        
        Parameters:
        -----------
        window_size : int, default=5
            Window size for smoothing
        min_contact_duration : int, default=3
            Minimum contact duration (not used in simplified algorithm)
        velocity_threshold : float, default=10
            Velocity threshold (not used in simplified algorithm)
        prominence_percentage : float, default=0.05
            Prominence percentage for peak detection
        auto_detect_errors : bool, default=True
            If True, automatically detect pose errors and create error_mask.
            If False, error_mask must be provided in extract() kwargs.
        error_detection_kwargs : dict, optional
            Keyword arguments for error detection if auto_detect_errors=True.
            See detect_pose_errors() for available options.
        show_plots : bool, default=False
            Whether to show plots during extraction
        """
        # self.window_size = window_size
        # self.min_contact_duration = min_contact_duration
        # self.velocity_threshold = velocity_threshold
        # self.prominence_percentage = prominence_percentage
        # self.show_plots = show_plots
        self._name = name or "zeni"
    
    @property
    def name(self) -> str:
        """Return the name of this extractor."""
        return self._name
    
    def extract(
        self, 
        postprocessed_data: pd.DataFrame, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract foot strikes using the Zeni algorithm.
        
        Parameters:
        -----------
        postprocessed_data : pd.DataFrame
            Postprocessed pose data (should include relative positions)
        **kwargs
            Additional parameters:
            - error_mask : pd.DataFrame, optional
                Error mask DataFrame. Required if auto_detect_errors=False.
            - scorer : str, optional
            - individual : str, optional
            - Other parameters override instance defaults
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'strikes': dict mapping paw names to lists of frame indices
            - 'figs': dict mapping paw names to matplotlib figures (if show_plots=False)
            - 'metadata': dict with algorithm parameters
        """
        # Get parameters (kwargs override instance defaults)
        # window_size = kwargs.pop('window_size', self.window_size)
        # min_contact_duration = kwargs.pop('min_contact_duration', self.min_contact_duration)
        # velocity_threshold = kwargs.pop('velocity_threshold', self.velocity_threshold)
        # prominence_percentage = kwargs.pop('prominence_percentage', self.prominence_percentage)
        # show_plots = kwargs.pop('show_plots', self.show_plots)
        
        
        # Run Zeni algorithm
        strikes = zeni_algorithm(
            postprocessed_data,
            **kwargs

            # window_size=window_size,
            # min_contact_duration=min_contact_duration,
            # velocity_threshold=velocity_threshold,
            # prominence_percentage=prominence_percentage,
            # show_plots=show_plots
        )
        
        return {
            'strikes': strikes,
            'metadata': {
                # 'window_size': window_size,
                # 'min_contact_duration': min_contact_duration,
                # 'velocity_threshold': velocity_threshold,
                # 'prominence_percentage': prominence_percentage,
            }
        }