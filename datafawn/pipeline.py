"""
Event Detection Pipeline

A modular pipeline for pose estimation, postprocessing, and event extraction.
Follows SOLID principles with interchangeable components.
"""


from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd


# =============== BASE INTERFACES (Strategy Pattern) =============== #

class PoseEstimator(ABC):
    """Abstract base class for pose estimation steps."""
    
    @abstractmethod
    def estimate(
        self, 
        video_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run pose estimation on a video.
        
        Parameters:
        -----------
        video_path : str or Path
            Path to the video file
        output_path : str or Path, optional
            Directory to save pose estimation outputs (labeled video, h5 file, etc.)
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


class SoundScapeGenerator(ABC):
    """Abstract base class for soundscape generation."""
    
    @abstractmethod
    def generate(
        self, 
        input_video_path: Union[str, Path], 
        events: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Generate a soundscape from events and input video.
        
        Parameters:
        -----------
        input_video_path : Union[str, Path]
            Path to the input video file.
        events : Dict[str, Any]
            Dictionary containing events.
            Structure: {(scorer, individual): {event_type: [frame_numbers]}}
        output_path : str or Path, optional
            Path for the output video file.
        **kwargs
            Additional parameters for soundscape generation
            
        Returns:
        --------
        str
            Path to the output video file
        """
        pass
    

# =============== PIPELINE CLASS =============== #

class EventDetectionPipeline:
    """
    Modular pipeline for event detection from videos or pose data.
    
    The pipeline consists of four stages:
    1. Pose Estimation (optional): Run pose estimation on a video
    2. Postprocessing: Apply transformations to pose data
    3. Event Extraction: Run one or more event extraction algorithms
    4. Soundscape Generation: Create audio overlays from events
    
    Each stage is modular and replaceable, following the Strategy pattern.
    You can start from any stage by providing the appropriate input data.
    """
    
    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        postprocessors: Optional[List[Postprocessor]] = None,
        event_extractors: Optional[List[EventExtractor]] = None,
        soundscape_generators: Optional[List[SoundScapeGenerator]] = None,
    ):
        """
        Initialize the event detection pipeline.
        
        Parameters:
        -----------
        pose_estimator : PoseEstimator, optional
            Pose estimation step. If None, pose data must be provided.
        postprocessors : list of Postprocessor, optional
            List of postprocessing steps to apply in sequence.
            If None, no postprocessing is applied.
        event_extractors : list of EventExtractor, optional
            List of event extraction algorithms to run.
            All extractors receive the same postprocessed data.
            If None, no event extraction is performed.
        soundscape_generators : list of SoundScapeGenerator, optional
            List of soundscape generation algorithms to run.
            If None, no soundscape generation is performed.
        """
        self.pose_estimator = pose_estimator
        self.postprocessors = postprocessors or []
        self.event_extractors = event_extractors or []
        self.soundscape_generators = soundscape_generators or []

        # Store intermediate results
        self._pose_data: Optional[pd.DataFrame] = None
        self._postprocessed_data: Optional[pd.DataFrame] = None
        self._event_results: Dict[str, Any] = {}
        self._soundscape_results: Dict[str, Any] = {}

    def run(
        self,
        # Stage 0: Video input (runs all stages)
        video_path: Optional[Union[str, Path]] = None,
        
        # Stage 1: Pose data input (skips pose estimation)
        pose_data: Optional[pd.DataFrame] = None,
        pose_data_path: Optional[Union[str, Path]] = None,
        
        # Stage 2: Postprocessed data input (skips pose estimation + postprocessing)
        postprocessed_data: Optional[pd.DataFrame] = None,
        postprocessed_data_path: Optional[Union[str, Path]] = None,
        
        # Stage 3: Events input (skips to soundscape generation)
        events: Optional[Dict[str, Any]] = None,
        events_path: Optional[Union[str, Path]] = None,
        
        # Output directory (organized structure)
        output_dir: Optional[Union[str, Path]] = None,
        
        # Video for soundscape (required for soundscape generation)
        soundscape_input_video: Optional[Union[Literal["raw", "pose_est"], str, Path]] = None,

        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        The pipeline auto-detects which stages to run based on the inputs provided.
        Provide the highest-level input you have to skip earlier stages.
        
        Priority order (highest to lowest):
        1. events/events_path -> skip to soundscape generation
        2. postprocessed_data/postprocessed_data_path -> skip to event extraction
        3. pose_data/pose_data_path -> skip to postprocessing
        4. video_path -> run full pipeline
        
        Parameters:
        -----------
        video_path : str or Path, optional
            Path to video file. Runs full pipeline including pose estimation.
        pose_data : pd.DataFrame, optional
            Pose data as DataFrame. Skips pose estimation.
        pose_data_path : str or Path, optional
            Path to pose data file (.h5). Skips pose estimation.
        postprocessed_data : pd.DataFrame, optional
            Already postprocessed data. Skips pose estimation and postprocessing.
        postprocessed_data_path : str or Path, optional
            Path to postprocessed data file (.h5). Skips pose estimation and postprocessing.
        events : Dict, optional
            Already extracted events. Skips to soundscape generation only.
        events_path : str or Path, optional
            Path to events file (.json). Skips to soundscape generation only.
        output_dir : str or Path, optional
            Directory for organized output. Creates subdirectories for each stage.
            If None, outputs are not saved to disk.
        soundscape_input_video : str, Path, or Literal["raw", "pose_est"], optional
            Video to use for soundscape generation.
            "raw" = original video_path
            "pose_est" = labeled video from pose estimation
            Or provide a direct path to any video file.
        **kwargs
            Additional keyword arguments passed to pipeline components.
            Use prefixes: 'pose_', 'postproc_', 'extract_', 'soundscape_'
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'pose_data': Raw pose data (pd.DataFrame or None)
            - 'postprocessed_data': Postprocessed data (pd.DataFrame or None)
            - 'events': Extracted events dict
            - 'soundscapes': Soundscape results dict
            - 'output_paths': Dict of paths to saved files (if output_dir provided)
            - 'metadata': Pipeline execution metadata
        """
        # Setup output directory structure
        output_paths = {}
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_paths = self._setup_output_dirs(output_dir)
        
        # Determine which stages to run based on inputs
        run_pose_estimation = False
        run_postprocessing = False
        run_event_extraction = False
        run_soundscape = len(self.soundscape_generators) > 0
        
        # Resolve input video path for soundscape
        soundscape_video_path = None
        pose_est_video_path = None
        
        # === STAGE DETECTION ===
        # Priority: events > postprocessed_data > pose_data > video_path
        
        if events is not None or events_path is not None:
            # Start from events - only run soundscape
            if events is not None:
                self._event_results = events
            else:
                self._event_results = self._load_events(events_path)
            self._pose_data = None
            self._postprocessed_data = None
            
        elif postprocessed_data is not None or postprocessed_data_path is not None:
            # Start from postprocessed data - run event extraction + soundscape
            if postprocessed_data is not None:
                self._postprocessed_data = postprocessed_data.copy()
            else:
                self._postprocessed_data = self._load_dataframe(postprocessed_data_path)
            self._pose_data = None
            run_event_extraction = True
            
        elif pose_data is not None or pose_data_path is not None:
            # Start from pose data - run postprocessing + event extraction + soundscape
            if pose_data is not None:
                self._pose_data = pose_data.copy()
            else:
                self._pose_data = self._load_dataframe(pose_data_path)
            run_postprocessing = True
            run_event_extraction = True
            
        elif video_path is not None:
            # Start from video - run full pipeline
            if self.pose_estimator is None:
                raise ValueError(
                    "video_path provided but no pose_estimator set. "
                    "Either provide pose_estimator or use pose_data/pose_data_path."
                )
            run_pose_estimation = True
            run_postprocessing = True
            run_event_extraction = True
        else:
            raise ValueError(
                "Must provide one of: video_path, pose_data, pose_data_path, "
                "postprocessed_data, postprocessed_data_path, events, or events_path"
            )
        
        # === EXECUTE STAGES ===
        
        # Stage 1: Pose Estimation
        if run_pose_estimation:
            pose_output_dir = output_paths.get('pose_estimation')
            pose_kwargs = self._filter_kwargs(kwargs, prefix='pose_')
            
            self._pose_data = self.pose_estimator.estimate(
                video_path,
                output_path=pose_output_dir,
                **pose_kwargs
            )
            
            # Find the generated labeled video
            if pose_output_dir is not None:
                pose_est_video_path = self._find_pose_est_video(video_path, pose_output_dir)
            
            # Save pose data
            if output_paths.get('pose_estimation'):
                pose_data_file = output_paths['pose_estimation'] / 'pose_data.h5'
                self._pose_data.to_hdf(pose_data_file, key='pose_data')
                output_paths['pose_data_file'] = pose_data_file
                print(f"📁 Saved pose data: {pose_data_file}")
        
        # Stage 2: Postprocessing
        if run_postprocessing:
            self._postprocessed_data = self._pose_data.copy()
            for postprocessor in self.postprocessors:
                postproc_kwargs = self._filter_kwargs(kwargs, prefix='postproc_')
                self._postprocessed_data = postprocessor.process(
                    self._postprocessed_data, **postproc_kwargs
                )
            
            # Save postprocessed data
            if output_paths.get('postprocessing'):
                postproc_file = output_paths['postprocessing'] / 'postprocessed_data.h5'
                self._postprocessed_data.to_hdf(postproc_file, key='postprocessed_data')
                output_paths['postprocessed_data_file'] = postproc_file
                print(f"📁 Saved postprocessed data: {postproc_file}")
        
        # Stage 3: Event Extraction
        if run_event_extraction:
            all_events = defaultdict(lambda: defaultdict(list))
            
            for extractor in self.event_extractors:
                extract_kwargs = self._filter_kwargs(kwargs, prefix='extract_')
                result = extractor.extract(self._postprocessed_data, **extract_kwargs)
                extracted_events = result.get('events', {})
                
                for (scorer, individual), event_dict in extracted_events.items():
                    for event_type, frame_numbers in event_dict.items():
                        all_events[(scorer, individual)][event_type].extend(frame_numbers)
            
            # Convert to regular dict and deduplicate/sort
            self._event_results = {}
            for (scorer, individual), event_dict in all_events.items():
                self._event_results[(scorer, individual)] = {
                    event_type: sorted(set(frame_numbers))
                    for event_type, frame_numbers in event_dict.items()
                }
            
            # Save events
            if output_paths.get('events'):
                events_file = output_paths['events'] / 'events.json'
                self._save_events(self._event_results, events_file)
                output_paths['events_file'] = events_file
                print(f"📁 Saved events: {events_file}")
        
        # Stage 4: Soundscape Generation
        self._soundscape_results = {}
        if run_soundscape and self._event_results:
            # Resolve soundscape input video
            soundscape_video_path = self._resolve_soundscape_video(
                soundscape_input_video,
                video_path,
                pose_est_video_path,
                output_paths.get('pose_estimation')
            )
            
            if soundscape_video_path is not None:
                soundscape_output = output_paths.get('soundscapes')
                
                for generator in self.soundscape_generators:
                    soundscape_kwargs = self._filter_kwargs(kwargs, prefix='soundscape_')
                    
                    # Determine output path for this generator
                    gen_name = getattr(generator, 'name', generator.__class__.__name__)
                    gen_output_path = None
                    if soundscape_output:
                        gen_output_path = soundscape_output / f'{gen_name}_output.mp4'
                    
                    # Pass events structure with individuals intact
                    # Structure: {(scorer, individual): {event_type: [frames]}}
                    result = generator.generate(
                        soundscape_video_path,
                        self._event_results,
                        output_path=gen_output_path,
                        **soundscape_kwargs
                    )
                    self._soundscape_results[gen_name] = result
                    
                    if gen_output_path:
                        output_paths[f'soundscape_{gen_name}'] = gen_output_path
                        print(f"📁 Saved soundscape: {gen_output_path}")
        
        # Return results
        return {
            'pose_data': self._pose_data,
            'postprocessed_data': self._postprocessed_data,
            'events': self._event_results,
            'soundscapes': self._soundscape_results,
            'output_paths': output_paths,
            'metadata': {
                'stages_run': {
                    'pose_estimation': run_pose_estimation,
                    'postprocessing': run_postprocessing,
                    'event_extraction': run_event_extraction,
                    'soundscape': run_soundscape and bool(self._soundscape_results),
                },
                'n_postprocessors': len(self.postprocessors),
                'n_extractors': len(self.event_extractors),
                'extractor_names': [e.name for e in self.event_extractors],
            }
        }
    
    def run_batch(
        self,
        video_paths: List[Union[str, Path]],
        output_base_dir: Union[str, Path],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline on multiple videos.
        
        Parameters:
        -----------
        video_paths : list of str or Path
            List of video file paths to process
        output_base_dir : str or Path
            Base directory for outputs. Each video gets a subdirectory.
        **kwargs
            Additional arguments passed to run() for each video
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of results from each pipeline run
        """
        output_base_dir = Path(output_base_dir)
        results = []
        
        for video_path in video_paths:
            video_path = Path(video_path)
            video_output_dir = output_base_dir / video_path.stem
            
            print(f"\n{'='*60}")
            print(f"Processing: {video_path.name}")
            print(f"Output: {video_output_dir}")
            print(f"{'='*60}")
            
            try:
                result = self.run(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    **kwargs
                )
                result['video_path'] = video_path
                result['success'] = True
            except Exception as e:
                print(f"ERROR processing {video_path.name}: {e}")
                result = {
                    'video_path': video_path,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n{'='*60}")
        print(f"Batch complete: {successful}/{len(video_paths)} successful")
        print(f"{'='*60}")
        
        return results
    
    # =============== SERIALIZATION METHODS =============== #
    
    def save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Save pipeline results to files.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from run()
        output_dir : str or Path
            Directory to save files
            
        Returns:
        --------
        Dict[str, Path]
            Mapping of result type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        
        # Save pose data
        if results.get('pose_data') is not None:
            pose_file = output_dir / 'pose_data.h5'
            results['pose_data'].to_hdf(pose_file, key='pose_data')
            saved_paths['pose_data'] = pose_file
            print(f"📁 Saved pose data: {pose_file}")
        
        # Save postprocessed data
        if results.get('postprocessed_data') is not None:
            postproc_file = output_dir / 'postprocessed_data.h5'
            results['postprocessed_data'].to_hdf(postproc_file, key='postprocessed_data')
            saved_paths['postprocessed_data'] = postproc_file
            print(f"📁 Saved postprocessed data: {postproc_file}")
        
        # Save events
        if results.get('events'):
            events_file = output_dir / 'events.json'
            self._save_events(results['events'], events_file)
            saved_paths['events'] = events_file
            print(f"📁 Saved events: {events_file}")
        
        # Save metadata
        if results.get('metadata'):
            metadata_file = output_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(results['metadata'], f, indent=2)
            saved_paths['metadata'] = metadata_file
            print(f"📁 Saved metadata: {metadata_file}")
        
        return saved_paths
    
    @staticmethod
    def load_results(output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Load previously saved pipeline results.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory containing saved results
            
        Returns:
        --------
        Dict[str, Any]
            Loaded results dictionary
        """
        output_dir = Path(output_dir)
        results = {}
        
        # Load pose data
        pose_file = output_dir / 'pose_data.h5'
        if pose_file.exists():
            results['pose_data'] = pd.read_hdf(pose_file)
        
        # Load postprocessed data
        postproc_file = output_dir / 'postprocessed_data.h5'
        if postproc_file.exists():
            results['postprocessed_data'] = pd.read_hdf(postproc_file)
        
        # Load events
        events_file = output_dir / 'events.json'
        if events_file.exists():
            results['events'] = EventDetectionPipeline._load_events_static(events_file)
        
        # Load metadata
        metadata_file = output_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                results['metadata'] = json.load(f)
        
        return results
    
    # =============== HELPER METHODS =============== #
    
    def _setup_output_dirs(self, output_dir: Path) -> Dict[str, Path]:
        """Create organized output directory structure."""
        subdirs = {
            'pose_estimation': output_dir / 'pose_estimation',
            'postprocessing': output_dir / 'postprocessing',
            'events': output_dir / 'events',
            'soundscapes': output_dir / 'soundscapes',
        }
        
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        return subdirs
    
    def _load_dataframe(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load DataFrame from file."""
        path = Path(path)
        
        if path.suffix == '.h5':
            return pd.read_hdf(path)
        elif path.suffix == '.csv':
            return pd.read_csv(path, index_col=0, header=[0, 1, 2, 3])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .h5 or .csv")
    
    def _load_events(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load events from JSON file."""
        return self._load_events_static(path)
    
    @staticmethod
    def _load_events_static(path: Union[str, Path]) -> Dict[str, Any]:
        """Load events from JSON file (static version)."""
        path = Path(path)
        with open(path, 'r') as f:
            raw_events = json.load(f)
        
        # Convert string keys back to tuples
        events = {}
        for key, value in raw_events.items():
            # Keys are stored as "scorer|individual"
            if '|' in key:
                scorer, individual = key.split('|', 1)
                events[(scorer, individual)] = value
            else:
                events[key] = value
        return events
    
    def _save_events(self, events: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save events to JSON file."""
        path = Path(path)
        
        # Convert tuple keys to strings and numpy types to native Python types
        serializable_events = {}
        for key, value in events.items():
            if isinstance(key, tuple):
                # Store as "scorer|individual"
                str_key = '|'.join(str(k) for k in key)
            else:
                str_key = str(key)
            
            # Convert nested dict with numpy int64 values to native Python ints
            if isinstance(value, dict):
                serializable_events[str_key] = {
                    event_type: [int(frame) for frame in frame_list]
                    for event_type, frame_list in value.items()
                }
            else:
                serializable_events[str_key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable_events, f, indent=2)
    
    def _resolve_soundscape_video(
        self,
        soundscape_input_video: Optional[Union[str, Path, Literal["raw", "pose_est"]]],
        video_path: Optional[Union[str, Path]],
        pose_est_video_path: Optional[Path],
        pose_est_dir: Optional[Path]
    ) -> Optional[Path]:
        """Resolve the video path to use for soundscape generation."""
        
        if soundscape_input_video is None:
            # Default to raw video if available
            if video_path is not None:
                return Path(video_path)
            return None
        
        if soundscape_input_video == "raw":
            if video_path is None:
                raise ValueError("soundscape_input_video='raw' but no video_path provided")
            return Path(video_path)
        
        if soundscape_input_video == "pose_est":
            # Try to find pose estimation video
            if pose_est_video_path is not None:
                return pose_est_video_path
            if pose_est_dir is not None and video_path is not None:
                found = self._find_pose_est_video(video_path, pose_est_dir)
                if found:
                    return found
            raise ValueError(
                "soundscape_input_video='pose_est' but no pose estimation video found. "
                "Run pose estimation first or provide a direct path."
            )
        
        # Direct path provided
        path = Path(soundscape_input_video)
        if not path.exists():
            raise FileNotFoundError(f"Soundscape input video not found: {path}")
        return path
    
    def _find_pose_est_video(
        self, 
        video_path: Optional[Union[str, Path]], 
        output_dir: Union[str, Path]
    ) -> Optional[Path]:
        """Find the labeled video file created by pose estimation."""
        output_dir = Path(output_dir)
        
        if not output_dir.exists():
            return None
        
        # DeepLabCut creates files like: {video_name}_{model_info}_labeled_before_adapt.mp4
        labeled_pattern = "*_labeled*.mp4"
        labeled_videos = list(output_dir.glob(labeled_pattern))
        
        if not labeled_videos:
            return None
        
        # If video_path provided, try to match by video stem
        if video_path is not None:
            video_path = Path(video_path)
            video_stem = video_path.stem
            matching = [v for v in labeled_videos if video_stem in v.stem]
            if matching:
                return matching[0]
        
        return labeled_videos[0]
    
    def _filter_kwargs(self, kwargs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Filter kwargs by prefix and remove the prefix."""
        return {
            key[len(prefix):]: value
            for key, value in kwargs.items()
            if key.startswith(prefix)
        }
    
    # =============== PROPERTIES =============== #
    
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
    
    @property
    def soundscape_results(self) -> Dict[str, Any]:
        """Get the soundscape results from the last run."""
        return self._soundscape_results
