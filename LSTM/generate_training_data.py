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
