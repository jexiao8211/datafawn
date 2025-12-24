"""
Concrete implementations for the event detection pipeline.

These classes wrap existing functions to make them compatible with the pipeline.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from datafawn.pipeline import PoseEstimator


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
            Additional parameters passed to deeplabcut.video_inference_superanimal.
            Note: Instance parameters (model_name, detector_name, etc.) cannot be overridden.
            
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
            print("No device specified, using auto-detection. Using: ", torch.cuda.get_device_name(0))
        else:
            device = torch.device(self.device)
        
        # Run inference with instance parameters
        processed_videos = deeplabcut.video_inference_superanimal(
            [video_path],
            self.model_name,
            max_individuals=self.max_individuals,
            model_name=self.hrnet_model,
            detector_name=self.detector_name,
            videotype=videotype,
            pcutoff=self.pcutoff,
            dest_folder=self.dest_folder,
            device=device,
            **kwargs
        )
        
        # Load the resulting H5 file
        # DeepLabCut creates files like: {video_name}_{model_info}.h5
        video_stem = video_path.stem
        h5_pattern = f"{video_stem}_{self.model_name}_{self.hrnet_model}_{self.detector_name}_*.h5"
        
        dest_path = Path(self.dest_folder)
        h5_files = list(dest_path.glob(h5_pattern))
        
        if not h5_files:
            raise FileNotFoundError(
                f"Could not find output H5 file matching pattern: {h5_pattern} "
                f"in {self.dest_folder}"
            )
        
        # Load the first matching file
        h5_file = h5_files[0]
        pose_data = pd.read_hdf(h5_file)
        pose_data = pose_data.sort_index(axis=1)
        
        return pose_data
