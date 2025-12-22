"""
Event Detection Pipeline

A modular pipeline for pose estimation, postprocessing, event extraction, and classification.
Follows SOLID principles with interchangeable components.
"""


from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np


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


class EventClassifier(ABC):
    """Abstract base class for event classification/refinement using ML models."""
    
    @abstractmethod
    def classify(
        self, 
        postprocessed_data: pd.DataFrame,
        event_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify or refine events using machine learning models.
        
        Parameters:
        -----------
        postprocessed_data : pd.DataFrame
            Postprocessed pose data
        event_results : Dict[str, Any]
            Results from event extractors (can be used as reference or ignored)
        **kwargs
            Additional parameters for classification
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing classification results.
            Format: {'events': [...], 'predictions': [...], 'metadata': {...}}
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this classifier."""
        pass


# =============== LSTM MODEL =============== #

class FawnLSTM(nn.Module):
    """LSTM model for sequence labeling (binary classification per frame)"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(FawnLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """Forward pass through LSTM"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        logits = self.fc(lstm_out)
        probs = torch.sigmoid(logits).squeeze(-1)
        return logits, probs


# =============== LSTM CLASSIFIER =============== #

class LSTMEventClassifier(EventClassifier):
    """
    LSTM-based event classifier that runs after event extraction.
    Can use Zeni results as training labels or run independently.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        sequence_length: Optional[int] = None,
        use_sliding_window: bool = True,
        stride: Optional[int] = None
    ):
        """
        Initialize LSTM event classifier.
        
        Args:
            model_path: Path to saved model weights (.pt or .pth file)
            input_dim: Input feature dimension (required if model_path is None)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            threshold: Classification threshold for binary prediction
            device: torch device (cuda/cpu)
            sequence_length: If set, processes data in fixed-length sequences
            use_sliding_window: If True, use sliding window approach for sequences
            stride: Stride for sliding window (defaults to sequence_length for non-overlapping)
        """
        self.model_path = Path(model_path) if model_path else None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.use_sliding_window = use_sliding_window
        self.stride = stride or (1 if use_sliding_window else sequence_length)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        if self.model_path and self.model_path.exists():
            self._load_model()
        elif input_dim is not None:
            self.model = FawnLSTM(input_dim, hidden_dim, num_layers, dropout)
            self.model.to(self.device)
            print(f"Initialized untrained LSTM model with input_dim={input_dim}")
        else:
            print("Warning: No model loaded. Call load_model() before classification.")
    
    def _load_model(self):
        """Load a trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.input_dim = checkpoint.get('input_dim', self.input_dim)
                self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
                self.num_layers = checkpoint.get('num_layers', self.num_layers)
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        if self.input_dim is None:
            raise ValueError("input_dim must be specified or saved in checkpoint")
        
        self.model = FawnLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded LSTM model from {self.model_path}")
    
    def load_model(self, model_path: Union[str, Path]):
        """Load a model from disk after initialization."""
        self.model_path = Path(model_path)
        self._load_model()
    
    def _prepare_features(self, postprocessed_data: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to feature array."""
        features = postprocessed_data.values
        
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)
        
        if self.input_dim is None:
            self.input_dim = features.shape[1]
        
        return features
    
    def _create_sequences(self, features: np.ndarray):
        """Create sequences from features using sliding window or chunking."""
        n_frames, n_features = features.shape
        
        if self.use_sliding_window:
            n_sequences = (n_frames - self.sequence_length) // self.stride + 1
            sequences = np.zeros((n_sequences, self.sequence_length, n_features))
            indices = []
            
            for i in range(n_sequences):
                start = i * self.stride
                end = start + self.sequence_length
                sequences[i] = features[start:end]
                indices.append(list(range(start, end)))
            
            return sequences, indices
        else:
            n_sequences = n_frames // self.sequence_length
            sequences = features[:n_sequences * self.sequence_length].reshape(
                n_sequences, self.sequence_length, n_features
            )
            indices = [list(range(i * self.sequence_length, (i + 1) * self.sequence_length)) 
                      for i in range(n_sequences)]
            
            return sequences, indices
    
    def _aggregate_predictions(self, all_predictions: list, all_probs: list, 
                               all_indices: list, n_frames: int):
        """Aggregate overlapping predictions using majority voting."""
        vote_counts = np.zeros(n_frames)
        prob_sums = np.zeros(n_frames)
        count_per_frame = np.zeros(n_frames)
        
        for preds, probs, indices in zip(all_predictions, all_probs, all_indices):
            for pred, prob, idx in zip(preds, probs, indices):
                vote_counts[idx] += pred
                prob_sums[idx] += prob
                count_per_frame[idx] += 1
        
        frame_probabilities = np.divide(
            prob_sums, count_per_frame, 
            out=np.zeros_like(prob_sums), 
            where=count_per_frame > 0
        )
        
        frame_predictions = (vote_counts > (count_per_frame / 2)).astype(int)
        
        return frame_predictions, frame_probabilities
    
    def classify(
        self, 
        postprocessed_data: pd.DataFrame,
        event_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify events using the trained LSTM model.
        
        Args:
            postprocessed_data: Postprocessed pose data
            event_results: Results from event extractors (e.g., Zeni)
            **kwargs: Additional parameters (threshold, sequence_length, etc.)
            
        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        threshold = kwargs.get('threshold', self.threshold)
        sequence_length = kwargs.get('sequence_length', self.sequence_length)
        
        features = self._prepare_features(postprocessed_data)
        n_frames = len(features)
        
        self.model.eval()
        with torch.no_grad():
            if sequence_length is not None:
                sequences, indices = self._create_sequences(features)
                
                all_predictions = []
                all_probs = []
                
                for seq in sequences:
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    _, probs = self.model(seq_tensor)
                    probs = probs.cpu().numpy()[0]
                    preds = (probs >= threshold).astype(int)
                    
                    all_predictions.append(preds)
                    all_probs.append(probs)
                
                frame_predictions, frame_probabilities = self._aggregate_predictions(
                    all_predictions, all_probs, indices, n_frames
                )
            else:
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                _, probs = self.model(features_tensor)
                
                frame_probabilities = probs.cpu().numpy()[0]
                frame_predictions = (frame_probabilities >= threshold).astype(int)
        
        strike_frames = np.where(frame_predictions == 1)[0].tolist()
        
        # Group consecutive frames into events
        events = []
        if strike_frames:
            current_event = {'start': strike_frames[0], 'end': strike_frames[0]}
            
            for frame in strike_frames[1:]:
                if frame == current_event['end'] + 1:
                    current_event['end'] = frame
                else:
                    events.append(current_event)
                    current_event = {'start': frame, 'end': frame}
            
            events.append(current_event)
        
        return {
            'events': events,
            'strike_frames': strike_frames,
            'frame_predictions': frame_predictions.tolist(),
            'frame_probabilities': frame_probabilities.tolist(),
            'n_strikes': len(strike_frames),
            'metadata': {
                'threshold': threshold,
                'sequence_length': sequence_length,
                'model_path': str(self.model_path) if self.model_path else None,
                'reference_extractor': list(event_results.keys())[0] if event_results else None
            }
        }
    
    @property
    def name(self) -> str:
        """Return classifier name."""
        return "LSTMEventClassifier"


# =============== PIPELINE CLASS =============== #

class EventDetectionPipeline:
    """
    Modular pipeline for event detection from videos or pose data.
    
    The pipeline consists of four stages:
    1. Pose Estimation (optional): Run pose estimation on a video
    2. Postprocessing: Apply transformations to pose data
    3. Event Extraction: Run one or more event extraction algorithms (e.g., Zeni)
    4. Classification (optional): Run ML-based classifiers (e.g., LSTM)
    
    Each stage is modular and replaceable, following the Strategy pattern.
    """
    
    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        postprocessors: Optional[List[Postprocessor]] = None,
        event_extractors: Optional[List[EventExtractor]] = None,
        classifiers: Optional[List[EventClassifier]] = None
    ):
        """
        Initialize the event detection pipeline.
        
        Parameters:
        -----------
        pose_estimator : PoseEstimator, optional
            Pose estimation step. If None, pose data must be provided via data_path.
        postprocessors : list of Postprocessor, optional
            List of postprocessing steps to apply in sequence.
        event_extractors : list of EventExtractor, optional
            List of event extraction algorithms to run (e.g., Zeni).
        classifiers : list of EventClassifier, optional
            List of ML classifiers to run after event extraction (e.g., LSTM).
        """
        self.pose_estimator = pose_estimator
        self.postprocessors = postprocessors or []
        self.event_extractors = event_extractors or []
        self.classifiers = classifiers or []
        
        # Store intermediate results
        self._pose_data: Optional[pd.DataFrame] = None
        self._postprocessed_data: Optional[pd.DataFrame] = None
        self._event_results: Dict[str, Any] = {}
        self._classification_results: Dict[str, Any] = {}
    
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
        pose_data : pd.DataFrame, optional
            Pose data as DataFrame. Takes precedence over pose_data_path.
        **kwargs
            Additional keyword arguments with prefixes:
            - 'pose_' for pose estimation
            - 'postproc_' for postprocessing
            - 'extract_' for event extraction
            - 'classify_' for classification
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'pose_data': Raw pose data (pd.DataFrame)
            - 'postprocessed_data': Postprocessed data (pd.DataFrame)
            - 'events': Dictionary mapping extractor names to their results
            - 'classifications': Dictionary mapping classifier names to their results
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
        
        # Step 4: Classification (NEW!)
        self._classification_results = {}
        for classifier in self.classifiers:
            classify_kwargs = self._filter_kwargs(kwargs, prefix='classify_')
            result = classifier.classify(
                self._postprocessed_data,
                self._event_results,
                **classify_kwargs
            )
            self._classification_results[classifier.name] = result
        
        # Return results
        return {
            'pose_data': self._pose_data,
            'postprocessed_data': self._postprocessed_data,
            'events': self._event_results,
            'classifications': self._classification_results,
            'metadata': {
                'n_postprocessors': len(self.postprocessors),
                'n_extractors': len(self.event_extractors),
                'n_classifiers': len(self.classifiers),
                'extractor_names': [e.name for e in self.event_extractors],
                'classifier_names': [c.name for c in self.classifiers]
            }
        }
    
    def _load_pose_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load pose data from file."""
        path = Path(path)
        
        if path.suffix == '.h5':
            return pd.read_hdf(path)
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
    
    @property
    def classification_results(self) -> Dict[str, Any]:
        """Get the classification results from the last run."""
        return self._classification_results