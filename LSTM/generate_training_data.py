import importlib
import torch
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datafawn.event_extractors import *
from datafawn.pipeline import *
from datafawn.pose_estimators import *
from datafawn.postprocessors import *

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")


DATA_PATH = "processed_vids/dog_running_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5"
VIDEO_PATH = "LSTM/dog_running.mp4"
OUTPUT_PATH = "training_data/lstm_training_data.npz"


# =============== POSE ESTIMATORS =============== #
dlc_estimator = DeepLabCutPoseEstimator(
    model_name='superanimal_quadruped',
    detector_name='fasterrcnn_resnet50_fpn_v2',
    hrnet_model='hrnet_w32',
    max_individuals=1,
    pcutoff=0.15,
    dest_folder='processed_vids',
    device=device
)

# =============== POSTPROCESSORS =============== #
paws = ['front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw']
reference_map = {
    'back_base': ['front_left_paw', 'front_right_paw'],
    'tail_base': ['back_left_paw', 'back_right_paw']
}
error_pp = ErrorPostprocessor(
    bodyparts=paws,
    use_velocity=True,
    use_likelihood=True,
    use_distance=True,
    velocity_kwargs={'threshold_pixels': 50, 'window_size': 5},
    likelihood_kwargs={'min_likelihood': 0.8},
    distance_kwargs={'reference_map': reference_map, 'max_distance': 300}
    )

rel_pp = RelativePawPositionPostprocessor()


# =============== EVENT EXTRACTORS =============== #
zeni_extractor = ZeniExtractor()

# Create pipeline
pipeline = EventDetectionPipeline(
    pose_estimator=dlc_estimator,
    postprocessors=[error_pp, rel_pp],
    event_extractors=[zeni_extractor]
)

# Run on existing pose data
results = pipeline.run(
    pose_data_path=DATA_PATH
)

print(results.keys())
print("Pose data shape:", results['pose_data'].shape)
print("Postprocessed data shape:", results['postprocessed_data'].shape)
print("Event extractors run:", results['metadata']['extractor_names'])
print(f"result events: {results['events']}")


def create_binary_labels_from_events(events, n_frames):
    """
    Convert event data to binary frame-level labels.
    
    Args:
        events: Event data from Zeni extractor (format depends on implementation)
        n_frames: Total number of frames in the sequence
    
    Returns:
        np.ndarray: Binary labels (n_frames,) where 1 = strike, 0 = no strike
    """
    labels = np.zeros(n_frames, dtype=np.int32)
    
    # Handle different possible event formats
    zeni_events = events.get('ZeniExtractor', events)
    
    # If events is a dict with 'events' key
    if isinstance(zeni_events, dict) and 'events' in zeni_events:
        event_list = zeni_events['events']
    else:
        event_list = zeni_events
    
    # Mark strike frames
    if event_list is not None:
        for event in event_list:
            if isinstance(event, dict):
                frame_idx = event.get('frame', event.get('frame_idx'))
                if frame_idx is not None and 0 <= frame_idx < n_frames:
                    labels[frame_idx] = 1
            elif isinstance(event, (int, np.integer)):
                if 0 <= event < n_frames:
                    labels[event] = 1
    
    return labels


def extract_feature_vectors(postprocessed_data):
    """
    Extract feature vectors from postprocessed pose data.
    
    Args:
        postprocessed_data: DataFrame with postprocessed pose data
    
    Returns:
        np.ndarray: Feature array of shape (n_frames, n_features)
    """
    # Convert DataFrame to numpy array
    # Remove any non-numeric columns and flatten the MultiIndex structure
    feature_array = postprocessed_data.values
    
    # If data has NaN values, handle them (e.g., forward fill)
    if np.any(np.isnan(feature_array)):
        print("Warning: NaN values detected in features. Filling with zeros.")
        feature_array = np.nan_to_num(feature_array, nan=0.0)
    
    print(f"Feature array shape: {feature_array.shape}")
    return feature_array

features = extract_feature_vectors(results['postprocessed_data'])
n_frames = len(features)
labels = create_binary_labels_from_events(results['events'], n_frames)


if np.sum(labels) == 0:
    print("\nWARNING: No strike frames detected!")
    print("Zeni events:", results['events'])
else:
    strike_frames = np.where(labels == 1)[0]
    print(f"Strike frame indices: {strike_frames}")


np.savez(
    OUTPUT_PATH,
    features=features,
    labels=labels,
    n_frames=n_frames,
    n_features=features.shape[1],
    strike_count=np.sum(labels)
)




def load_training_data(data_path, sequence_length=None):
    """
    Load training data and optionally create sequences.
    
    Args:
        data_path: Path to .npz file
        sequence_length: If provided, split data into fixed-length sequences
    
    Returns:
        dict with 'features', 'labels', and metadata
    """
    data = np.load(data_path)
    features = data['features']
    labels = data['labels']
    
    if sequence_length is not None:
        # Create overlapping or non-overlapping sequences
        n_sequences = len(features) // sequence_length
        features = features[:n_sequences * sequence_length].reshape(
            n_sequences, sequence_length, -1
        )
        labels = labels[:n_sequences * sequence_length].reshape(
            n_sequences, sequence_length
        )
    
    return {
        'features': features,
        'labels': labels,
        'n_frames': data['n_frames'],
        'n_features': data['n_features'],
        'strike_count': data['strike_count']
    }

loaded_data = load_training_data(OUTPUT_PATH)
print(f"Loaded features shape: {loaded_data['features'].shape}")
print(f"Loaded labels shape: {loaded_data['labels'].shape}")

# Example with sequences
print("\n=== Example: Creating 30-frame sequences ===")
seq_data = load_training_data(OUTPUT_PATH, sequence_length=30)
print(f"Sequence features shape: {seq_data['features'].shape}")
print(f"Sequence labels shape: {seq_data['labels'].shape}")