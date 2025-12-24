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
        smooth_window_size: int = 5,
        prominence_percentage: float = 0.05,
        orientation_likelihood_threshold: float = 0.0,
        orientation_smooth_window_size: int = 15,
        show_plots: bool = False,
        name: Optional[str] = None,
    ):
        """
        Initialize the Zeni extractor.
        
        Parameters:
        -----------
        window_size : int, default=5
            Window size for smoothing
        prominence_percentage : float, default=0.05
            Prominence percentage for peak detection
        orientation_likelihood_threshold : float, default=0.0
            Minimum likelihood for orientation data to be considered valid.
        orientation_smooth_window_size : int, default=15
            Window size for smoothing orientation when determining direction of motion.
        show_plots : bool, default=False
            Whether to show plots during extraction
        name : str, optional
            Name of the extractor. Default is "zeni".
        """
        self.smooth_window_size = smooth_window_size
        self.prominence_percentage = prominence_percentage
        self.orientation_likelihood_threshold = orientation_likelihood_threshold
        self.orientation_smooth_window_size = orientation_smooth_window_size
        self.show_plots = show_plots
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
            Ignored. All parameters are set at initialization.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'events': dict mapping (scorer, individual) to event dictionaries
            - 'metadata': dict with algorithm parameters
        """
        # Run Zeni algorithm with instance parameters
        strikes = zeni_algorithm(
            postprocessed_data,
            smooth_window_size=self.smooth_window_size,
            prominence_percentage=self.prominence_percentage,
            orientation_likelihood_threshold=self.orientation_likelihood_threshold,
            orientation_smooth_window_size=self.orientation_smooth_window_size,
            show_plots=self.show_plots
        )
        
        return {
            'events': strikes,
            'metadata': {
                'smooth_window_size': self.smooth_window_size,
                'prominence_percentage': self.prominence_percentage,
                'orientation_likelihood_threshold': self.orientation_likelihood_threshold,
                'orientation_smooth_window_size': self.orientation_smooth_window_size,
                'show_plots': self.show_plots
            }
        }