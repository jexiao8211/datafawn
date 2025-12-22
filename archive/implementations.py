"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from event_detection.pipeline import PoseEstimator, Postprocessor, EventExtractor
from utils.postprocessing import paw_to_relative_position, detect_pose_errors
from event_extraction.zeni import zeni_algorithm_vis






# =============== EVENT EXTRACTORS =============== #

class ZeniExtractor(EventExtractor):
    """
    Event extractor using the Zeni algorithm for foot strike detection.
    
    Wraps the zeni_algorithm_vis function from event_extraction.zeni.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        min_contact_duration: int = 3,
        velocity_threshold: float = 10,
        prominence_percentage: float = 0.05,
        auto_detect_errors: bool = True,
        error_detection_kwargs: Optional[Dict[str, Any]] = None,
        show_plots: bool = False,
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
        self.window_size = window_size
        self.min_contact_duration = min_contact_duration
        self.velocity_threshold = velocity_threshold
        self.prominence_percentage = prominence_percentage
        self.auto_detect_errors = auto_detect_errors
        self.error_detection_kwargs = error_detection_kwargs or {}
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
        window_size = kwargs.pop('window_size', self.window_size)
        min_contact_duration = kwargs.pop('min_contact_duration', self.min_contact_duration)
        velocity_threshold = kwargs.pop('velocity_threshold', self.velocity_threshold)
        prominence_percentage = kwargs.pop('prominence_percentage', self.prominence_percentage)
        scorer = kwargs.pop('scorer', None)
        individual = kwargs.pop('individual', None)
        show_plots = kwargs.pop('show_plots', self.show_plots)
        
        # Handle error mask
        error_mask = kwargs.pop('error_mask', None)
        if error_mask is None and self.auto_detect_errors:
            # Auto-detect errors
            paws = ['front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw']
            error_detection_kwargs = {
                'bodyparts': paws,
                'velocity_threshold': 50,
                'min_likelihood': 0.0,
                'max_distance': 300,
                **self.error_detection_kwargs
            }
            
            # Infer scorer and individual if not provided
            if scorer is None:
                scorer = postprocessed_data.columns.get_level_values(0).unique()[0]
            if individual is None:
                individual = postprocessed_data.columns.get_level_values(1).unique()[0]
            
            error_detection_kwargs['scorer'] = scorer
            error_detection_kwargs['individual'] = individual
            
            # Get original pose data (before relative positions were added)
            # Filter out _rel columns to get original bodyparts
            original_bodyparts = [
                bp for bp in postprocessed_data.columns.get_level_values('bodyparts').unique()
                if not bp.endswith('_rel')
            ]
            
            if original_bodyparts:
                # Reconstruct original data structure by selecting non-_rel columns
                original_cols = []
                for bp in original_bodyparts:
                    for coord in ['x', 'y', 'likelihood']:
                        col_tuple = (scorer, individual, bp, coord)
                        if col_tuple in postprocessed_data.columns:
                            original_cols.append(col_tuple)
                
                if original_cols:
                    original_data = postprocessed_data[original_cols]
                    error_details = detect_pose_errors(original_data, **error_detection_kwargs)
                    
                    # Create error_mask in the format expected by zeni_algorithm
                    error_mask = pd.DataFrame(index=error_details.index)
                    for paw in paws:
                        error_col = f'{paw}_error'
                        if error_col in error_details.columns:
                            error_mask[paw] = error_details[error_col]
                        else:
                            error_mask[paw] = False
        
        # Run Zeni algorithm
        strikes, figs = zeni_algorithm_vis(
            postprocessed_data,
            window_size=window_size,
            scorer=scorer,
            individual=individual,
            min_contact_duration=min_contact_duration,
            velocity_threshold=velocity_threshold,
            error_mask=error_mask,
            prominence_percentage=prominence_percentage,
            show_plots=show_plots
        )
        
        return {
            'strikes': strikes,
            'figs': figs if not show_plots else {},
            'metadata': {
                'window_size': window_size,
                'min_contact_duration': min_contact_duration,
                'velocity_threshold': velocity_threshold,
                'prominence_percentage': prominence_percentage,
                'auto_detected_errors': self.auto_detect_errors and error_mask is not None
            }
        }



