"""
Event Detection Pipeline

A modular pipeline for pose estimation, postprocessing, and event extraction.
Follows SOLID principles with interchangeable components.
"""


from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
import pandas as pd


# =============== BASE INTERFACES (Strategy Pattern) =============== #

class PoseEstimator(ABC):
    """Abstract base class for pose estimation steps."""
    
    @abstractmethod
    def estimate(self, video_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Run pose estimation on a video.
        
        Parameters:
        -----------
        video_path : str or Path
            Path to the video file
        **kwargs
            Additional parameters for the pose estimator
            
        Returns:
        --------
        pd.DataFrame
            Pose data with MultiIndex columns (scorer, individual, bodypart, coords)
        """
        pass


class Postprocessor(ABC):
    """Abstract base class for postprocessing steps."""
    
    @abstractmethod
    def process(self, pose_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply postprocessing transformations to pose data.
        
        Parameters:
        -----------
        pose_data : pd.DataFrame
            Raw pose data from pose estimation
        **kwargs
            Additional parameters for postprocessing
            
        Returns:
        --------
        pd.DataFrame
            Postprocessed pose data
        """
        pass


class EventExtractor(ABC):
    """Abstract base class for event extraction algorithms."""
    
    @abstractmethod
    def extract(self, postprocessed_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Extract events from postprocessed pose data.
        
        Parameters:
        -----------
        postprocessed_data : pd.DataFrame
            Postprocessed pose data
        **kwargs
            Additional parameters for event extraction
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing extracted events. Structure is up to the implementation.
            Common format: {'events': [...], 'metadata': {...}}
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this extractor."""
        pass


# =============== PIPELINE CLASS =============== #

class EventDetectionPipeline:
    """
    Modular pipeline for event detection from videos or pose data.
    
    The pipeline consists of three stages:
    1. Pose Estimation (optional): Run pose estimation on a video
    2. Postprocessing: Apply transformations to pose data
    3. Event Extraction: Run one or more event extraction algorithms
    
    Each stage is modular and replaceable, following the Strategy pattern.
    """
    
    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        postprocessors: Optional[List[Postprocessor]] = None,
        event_extractors: Optional[List[EventExtractor]] = None
    ):
        """
        Initialize the event detection pipeline.
        
        Parameters:
        -----------
        pose_estimator : PoseEstimator, optional
            Pose estimation step. If None, pose data must be provided via data_path.
        postprocessors : list of Postprocessor, optional
            List of postprocessing steps to apply in sequence.
            If None, no postprocessing is applied.
        event_extractors : list of EventExtractor, optional
            List of event extraction algorithms to run.
            All extractors receive the same postprocessed data.
            If None, no event extraction is performed.
        """
        self.pose_estimator = pose_estimator
        self.postprocessors = postprocessors or []
        self.event_extractors = event_extractors or []
        
        # Store intermediate results
        self._pose_data: Optional[pd.DataFrame] = None
        self._postprocessed_data: Optional[pd.DataFrame] = None
        self._event_results: Dict[str, Any] = {}
    
    def run(
        self,
        video_path: Optional[Union[str, Path]] = None,
        pose_data_path: Optional[Union[str, Path]] = None,
        pose_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Parameters:
        -----------
        video_path : str or Path, optional
            Path to video file. Requires pose_estimator to be set.
        pose_data_path : str or Path, optional
            Path to existing pose data file (e.g., .h5, .csv).
            Will be loaded using pandas.
        pose_data : pd.DataFrame, optional
            Pose data as DataFrame. Takes precedence over pose_data_path.
        **kwargs
            Additional keyword arguments passed to:
            - pose_estimator.estimate() if video_path is provided
            - postprocessors.process() for each postprocessor
            - event_extractors.extract() for each extractor
            
            Use prefixes to target specific steps:
            - 'pose_' prefix for pose estimation kwargs
            - 'postproc_' prefix for postprocessing kwargs
            - 'extract_' prefix for event extraction kwargs
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'pose_data': Raw pose data (pd.DataFrame)
            - 'postprocessed_data': Postprocessed data (pd.DataFrame)
            - 'events': Dictionary mapping extractor names to their results
            - 'metadata': Pipeline execution metadata
        """
        # Step 1: Pose Estimation
        if pose_data is not None:
            self._pose_data = pose_data.copy()
        elif pose_data_path is not None:
            self._pose_data = self._load_pose_data(pose_data_path)
        elif video_path is not None:
            if self.pose_estimator is None:
                raise ValueError(
                    "video_path provided but no pose_estimator set. "
                    "Either provide pose_estimator or use pose_data_path/pose_data."
                )
            pose_kwargs = self._filter_kwargs(kwargs, prefix='pose_')
            self._pose_data = self.pose_estimator.estimate(video_path, **pose_kwargs)
        else:
            raise ValueError(
                "Must provide one of: video_path, pose_data_path, or pose_data"
            )
        
        # Step 2: Postprocessing
        self._postprocessed_data = self._pose_data.copy()
        for postprocessor in self.postprocessors:
            postproc_kwargs = self._filter_kwargs(kwargs, prefix='postproc_')
            self._postprocessed_data = postprocessor.process(
                self._postprocessed_data, **postproc_kwargs
            )
        
        # Step 3: Event Extraction
        self._event_results = {}
        for extractor in self.event_extractors:
            extract_kwargs = self._filter_kwargs(kwargs, prefix='extract_')
            result = extractor.extract(self._postprocessed_data, **extract_kwargs)
            self._event_results[extractor.name] = result
        
        # Return results
        return {
            'pose_data': self._pose_data,
            'postprocessed_data': self._postprocessed_data,
            'events': self._event_results,
            'metadata': {
                'n_postprocessors': len(self.postprocessors),
                'n_extractors': len(self.event_extractors),
                'extractor_names': [e.name for e in self.event_extractors]
            }
        }
    
    def _load_pose_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load pose data from file."""
        path = Path(path)
        
        if path.suffix == '.h5':
            # DeepLabCut format
            return pd.read_hdf(path)
        # TODO: add support for other file formats?
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _filter_kwargs(self, kwargs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Filter kwargs by prefix and remove the prefix."""
        filtered = {}
        for key, value in kwargs.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                filtered[new_key] = value
        return filtered
    
    @property
    def pose_data(self) -> Optional[pd.DataFrame]:
        """Get the raw pose data from the last run."""
        return self._pose_data
    
    @property
    def postprocessed_data(self) -> Optional[pd.DataFrame]:
        """Get the postprocessed data from the last run."""
        return self._postprocessed_data
    
    @property
    def event_results(self) -> Dict[str, Any]:
        """Get the event extraction results from the last run."""
        return self._event_results

