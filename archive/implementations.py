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


# =============== POSE ESTIMATORS =============== #

class DeepLabCutPoseEstimator(PoseEstimator):
    """
    Pose estimator using DeepLabCut SuperAnimal models.
    
    Wraps deeplabcut.video_inference_superanimal.
    """
    
    def __init__(
        self,
        model_name: str = 'superanimal_quadruped',
        detector_name: str = 'fasterrcnn_resnet50_fpn_v2',
        hrnet_model: str = 'hrnet_w32',
        max_individuals: int = 1,
        pcutoff: float = 0.15,
        dest_folder: str = 'processed_vids',
        device: Optional[str] = None
    ):
        """
        Initialize the DeepLabCut pose estimator.
        
        Parameters:
        -----------
        model_name : str, default='superanimal_quadruped'
            SuperAnimal model name
        detector_name : str, default='fasterrcnn_resnet50_fpn_v2'
            Detector model name
        hrnet_model : str, default='hrnet_w32'
            HRNet model variant
        max_individuals : int, default=1
            Maximum number of individuals to track
        pcutoff : float, default=0.15
            Likelihood cutoff
        dest_folder : str, default='processed_vids'
            Destination folder for output files
        device : str, optional
            Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        self.model_name = model_name
        self.detector_name = detector_name
        self.hrnet_model = hrnet_model
        self.max_individuals = max_individuals
        self.pcutoff = pcutoff
        self.dest_folder = dest_folder
        self.device = device
    
    def estimate(self, video_path, **kwargs) -> pd.DataFrame:
        """
        Run pose estimation on a video.
        
        Parameters:
        -----------
        video_path : str or Path
            Path to video file
        **kwargs
            Additional parameters that override instance defaults
            
        Returns:
        --------
        pd.DataFrame
            Pose data with MultiIndex columns
        """
        import deeplabcut
        import torch
        from pathlib import Path
        
        video_path = Path(video_path)
        videotype = video_path.suffix
        
        # Get device
        if self.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.device)
        
        # Get parameters (kwargs override instance defaults)
        model_name = kwargs.pop('model_name', self.model_name)
        detector_name = kwargs.pop('detector_name', self.detector_name)
        hrnet_model = kwargs.pop('hrnet_model', self.hrnet_model)
        max_individuals = kwargs.pop('max_individuals', self.max_individuals)
        pcutoff = kwargs.pop('pcutoff', self.pcutoff)
        dest_folder = kwargs.pop('dest_folder', self.dest_folder)
        
        # Run inference
        processed_videos = deeplabcut.video_inference_superanimal(
            [video_path],
            model_name,
            max_individuals=max_individuals,
            model_name=hrnet_model,
            detector_name=detector_name,
            videotype=videotype,
            pcutoff=pcutoff,
            dest_folder=dest_folder,
            device=device,
            **kwargs
        )
        
        # Load the resulting H5 file
        # DeepLabCut creates files like: {video_name}_{model_info}.h5
        video_stem = video_path.stem
        h5_pattern = f"{video_stem}_{model_name}_{hrnet_model}_{detector_name}_*.h5"
        
        dest_path = Path(dest_folder)
        h5_files = list(dest_path.glob(h5_pattern))
        
        if not h5_files:
            raise FileNotFoundError(
                f"Could not find output H5 file matching pattern: {h5_pattern} "
                f"in {dest_folder}"
            )
        
        # Load the first matching file
        h5_file = h5_files[0]
        pose_data = pd.read_hdf(h5_file)
        pose_data = pose_data.sort_index(axis=1)
        
        return pose_data

