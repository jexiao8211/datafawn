"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from datafawn.pipeline import Postprocessor
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from postprocessing.error_mask import (
    detect_velocity_errors,
    detect_likelihood_errors,
    detect_distance_errors
)
from postprocessing.norm import paw_to_relative_position

class ErrorPostprocessor(Postprocessor):
    """
    Postprocessor that detects and corrects errors in pose data.
    
    Allows users to select which error detection methods to apply.
    Detected errors are replaced with NaN, then filled using forward fill
    followed by backward fill.
    """
    
    def __init__(
        self,
        bodyparts: List[str],
        use_velocity: bool = True,
        use_likelihood: bool = True,
        use_distance: bool = True,
        velocity_kwargs: Optional[Dict[str, Any]] = None,
        likelihood_kwargs: Optional[Dict[str, Any]] = None,
        distance_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ErrorPostprocessor.
        
        Parameters:
        -----------
        bodyparts : list of str
            List of bodypart names to check for errors
        use_velocity : bool, default=True
            Whether to use velocity-based error detection
        use_likelihood : bool, default=True
            Whether to use likelihood-based error detection
        use_distance : bool, default=True
            Whether to use distance-based error detection
        velocity_kwargs : dict, optional
            Additional keyword arguments for detect_velocity_errors
        likelihood_kwargs : dict, optional
            Additional keyword arguments for detect_likelihood_errors
        distance_kwargs : dict, optional
            Additional keyword arguments for detect_distance_errors
            Must include 'reference_map' if use_distance=True
        """
        self.bodyparts = bodyparts
        self.use_velocity = use_velocity
        self.use_likelihood = use_likelihood
        self.use_distance = use_distance
        self.velocity_kwargs = velocity_kwargs or {}
        self.likelihood_kwargs = likelihood_kwargs or {}
        self.distance_kwargs = distance_kwargs or {}
        
        if use_distance and 'reference_map' not in self.distance_kwargs:
            raise ValueError(
                "use_distance=True requires 'reference_map' in distance_kwargs"
            )
    
    def process(self, pose_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Detect and correct errors in pose data.
        
        Parameters:
        -----------
        pose_data : pd.DataFrame
            Pose data with MultiIndex columns (scorer, individual, bodypart, coords)
        **kwargs
            Additional parameters that override instance settings:
            - Any parameters for error detection functions
            
        Returns:
        --------
        pd.DataFrame
            Pose data with errors corrected (NaN values filled)
        """

        # Create a copy to avoid modifying the original
        corrected_data = pose_data.copy()
        
        # Collect all error masks
        error_masks = []
        
        # Run velocity error detection if enabled
        if self.use_velocity:
            velocity_kwargs = {**self.velocity_kwargs, **kwargs}
            velocity_kwargs.pop('scorer', None)  # Remove if present, will add back
            velocity_kwargs.pop('individual', None)
            velocity_mask = detect_velocity_errors(
                pose_data,
                self.bodyparts,
                **velocity_kwargs
            )
            error_masks.append(velocity_mask)
        
        # Run likelihood error detection if enabled
        if self.use_likelihood:
            likelihood_kwargs = {**self.likelihood_kwargs, **kwargs}
            likelihood_kwargs.pop('scorer', None)
            likelihood_kwargs.pop('individual', None)
            likelihood_mask = detect_likelihood_errors(
                pose_data,
                self.bodyparts,
                **likelihood_kwargs
            )
            error_masks.append(likelihood_mask)
        
        # Run distance error detection if enabled
        if self.use_distance:
            distance_kwargs = {**self.distance_kwargs, **kwargs}
            distance_kwargs.pop('scorer', None)
            distance_kwargs.pop('individual', None)
            distance_mask = detect_distance_errors(
                pose_data,
                self.bodyparts,
                **distance_kwargs
            )
            error_masks.append(distance_mask)
        
        # Combine all error masks (OR operation: if any mask says error, it's an error)
        if error_masks:
            # Get all unique columns from all masks
            all_columns = set()
            for mask in error_masks:
                all_columns.update(mask.columns)
            all_columns = sorted(all_columns)  # Sort for consistency
            
            # Initialize combined mask with all columns
            combined_mask = pd.DataFrame(
                False,
                index=pose_data.index,
                columns=pd.MultiIndex.from_tuples(all_columns, names=['scorer', 'individual', 'bodypart'])
            )
            
            # Combine all masks with OR operation
            for mask in error_masks:
                # Align mask to combined_mask structure and combine
                for col in mask.columns:
                    if col in combined_mask.columns:
                        # Reindex mask column to match combined_mask index if needed
                        mask_col = mask[col].reindex(combined_mask.index, fill_value=False)
                        combined_mask[col] = combined_mask[col] | mask_col
            
            # Apply the combined mask to replace errors with NaN
            # The mask has columns (scorer, individual, bodypart)
            # The data has columns (scorer, individual, bodypart, coords)
            # We need to set NaN for x, y, and likelihood when an error is detected
            
            for (scorer_name, individual_name, bodypart) in combined_mask.columns:
                if bodypart not in self.bodyparts:
                    continue
                
                # Get the error mask for this (scorer, individual, bodypart)
                error_series = combined_mask[(scorer_name, individual_name, bodypart)]
                
                # Set NaN for x, y, and likelihood coordinates where errors are detected
                for coord in ['x', 'y', 'likelihood']:
                    try:
                        col = (scorer_name, individual_name, bodypart, coord)
                        if col in corrected_data.columns:
                            corrected_data.loc[error_series, col] = np.nan
                    except (KeyError, IndexError):
                        continue
        
        # Apply forward fill, then backward fill
        # corrected_data = corrected_data.ffill()
        # corrected_data = corrected_data.bfill()
        
        return corrected_data


class RelativePawPositionPostprocessor(Postprocessor):
    """
    Postprocessor that adds relative position coordinates for paws.
    
    Wraps the paw_to_relative_position function from utils.postprocessing.
    """
    
    def process(self, pose_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Append relative paw position coordinates to pose data as new columns.
        
        Parameters:
        -----------
        pose_data : pd.DataFrame
            Raw pose data
        **kwargs
            Additional parameters passed to paw_to_relative_position:
            - scorer : str, optional
            - individual : str, optional
            - append_to_df : bool, default=True
            
        Returns:
        --------
        pd.DataFrame
            Pose data with relative position columns added
        """
        return paw_to_relative_position(pose_data, **kwargs)
