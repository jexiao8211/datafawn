"""Example: Run pipeline on existing pose data."""

# Deer running
EXAMPLE_DATA_PATH = "deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5"
EXAMPLE_VIDEO_PATH = 'processed_vids\\deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2__labeled_before_adapt.mp4'


# =============== POSE ESTIMATORS =============== #
dlc_estimator = datafawn.DeepLabCutPoseEstimator(
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
rel_paws = ['front_left_paw_rel', 'front_right_paw_rel', 'back_left_paw_rel', 'back_right_paw_rel']

reference_map = {
    'back_base': ['front_left_paw', 'front_right_paw'],
    'tail_base': ['back_left_paw', 'back_right_paw']
}

rel_pp = datafawn.RelativePawPositionPostprocessor()

error_pp = datafawn.ErrorPostprocessor(
    bodyparts=rel_paws,
    use_velocity=True,
    use_likelihood=True,
    use_distance=True,
    velocity_kwargs={'threshold_pixels': 50, 'window_size': 5},
    likelihood_kwargs={'min_likelihood': 0.5},
    distance_kwargs={'reference_map': reference_map, 'max_distance': 300}
    )



# =============== EVENT EXTRACTORS =============== #
zeni_extractor = datafawn.ZeniExtractor(
    smooth_window_size=5,
    prominence_percentage=0.08,
    orientation_likelihood_threshold=0.0,
    orientation_smooth_window_size=15,
    show_plots=True
)

# Create pipeline
pipeline = datafawn.EventDetectionPipeline(
    pose_estimator=dlc_estimator,
    postprocessors=[rel_pp, error_pp],
    event_extractors=[zeni_extractor]
)

# Run on existing pose data
results = pipeline.run(
    pose_data_path=EXAMPLE_DATA_PATH,
    # video_path=EXAMPLE_VIDEO_PATH,   # uncomment this to run DLC on raw video first
)