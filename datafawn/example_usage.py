"""
Example usage of the Event Detection Pipeline

This demonstrates how to use the modular pipeline system.
"""

from pathlib import Path
from event_detection import (
    EventDetectionPipeline,
    RelativePositionPostprocessor,
    ZeniExtractor,
    DeepLabCutPoseEstimator
)


def example_with_existing_pose_data():
    """Example: Run pipeline on existing pose data."""
    
    # Create pipeline components
    postprocessor = RelativePositionPostprocessor()
    zeni_extractor = ZeniExtractor(
        window_size=5,
        prominence_percentage=0.05,
        auto_detect_errors=True,
        show_plots=False
    )
    
    # Create pipeline
    pipeline = EventDetectionPipeline(
        postprocessors=[postprocessor],
        event_extractors=[zeni_extractor]
    )
    
    # Run on existing pose data
    results = pipeline.run(
        pose_data_path="processed_vids/deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5"
    )
    
    # Access results
    print("Pose data shape:", results['pose_data'].shape)
    print("Postprocessed data shape:", results['postprocessed_data'].shape)
    print("Event extractors run:", results['metadata']['extractor_names'])
    
    # Access Zeni results
    zeni_results = results['events']['zeni']
    strikes = zeni_results['strikes']
    print("\nFoot strikes detected:")
    for paw, frames in strikes.items():
        print(f"  {paw}: {len(frames)} strikes")
    
    return results


def example_with_video():
    """Example: Run pipeline on video (requires pose estimation)."""
    
    # Create pose estimator
    pose_estimator = DeepLabCutPoseEstimator(
        model_name='superanimal_quadruped',
        hrnet_model='hrnet_w32',
        detector_name='fasterrcnn_resnet50_fpn_v2',
        max_individuals=1,
        pcutoff=0.15,
        dest_folder='processed_vids'
    )
    
    # Create postprocessor
    postprocessor = RelativePositionPostprocessor()
    
    # Create event extractors
    zeni_extractor = ZeniExtractor(
        window_size=5,
        auto_detect_errors=True,
        show_plots=False
    )
    
    # Create pipeline
    pipeline = EventDetectionPipeline(
        pose_estimator=pose_estimator,
        postprocessors=[postprocessor],
        event_extractors=[zeni_extractor]
    )
    
    # Run on video
    results = pipeline.run(
        video_path="videos/deer2.mp4"
    )
    
    # Access results
    zeni_results = results['events']['zeni']
    strikes = zeni_results['strikes']
    print("\nFoot strikes detected:")
    for paw, frames in strikes.items():
        print(f"  {paw}: {len(frames)} strikes")
    
    return results


def example_multiple_extractors():
    """Example: Run multiple event extractors on the same data."""
    
    # Create postprocessor
    postprocessor = RelativePositionPostprocessor()
    
    # Create multiple extractors with different parameters
    zeni_extractor_1 = ZeniExtractor(
        window_size=5,
        prominence_percentage=0.05,
        auto_detect_errors=True,
        show_plots=False
    )
    
    zeni_extractor_2 = ZeniExtractor(
        window_size=7,  # Different window size
        prominence_percentage=0.03,  # Different prominence
        auto_detect_errors=True,
        show_plots=False
    )
    
    # Create pipeline with multiple extractors
    pipeline = EventDetectionPipeline(
        postprocessors=[postprocessor],
        event_extractors=[zeni_extractor_1, zeni_extractor_2]
    )
    
    # Run pipeline
    results = pipeline.run(
        pose_data_path="processed_vids/deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5"
    )
    
    # Access results from both extractors
    print("Results from extractor 1 (zeni):")
    strikes_1 = results['events']['zeni']['strikes']
    for paw, frames in strikes_1.items():
        print(f"  {paw}: {len(frames)} strikes")
    
    # Note: Both extractors have the same name "zeni", so the second one will overwrite the first
    # To avoid this, you'd need to create a custom extractor with a different name
    # or modify the ZeniExtractor to accept a name parameter
    
    return results


def example_custom_parameters():
    """Example: Pass custom parameters to pipeline steps."""
    
    pipeline = EventDetectionPipeline(
        postprocessors=[RelativePositionPostprocessor()],
        event_extractors=[ZeniExtractor(auto_detect_errors=True)]
    )
    
    # Pass parameters with prefixes
    results = pipeline.run(
        pose_data_path="processed_vids/deer2_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5",
        # Parameters for postprocessing (prefix: postproc_)
        postproc_scorer=None,  # Auto-detect
        postproc_individual=None,  # Auto-detect
        # Parameters for event extraction (prefix: extract_)
        extract_window_size=7,
        extract_prominence_percentage=0.03,
        extract_scorer=None,
        extract_individual=None
    )
    
    return results


if __name__ == "__main__":
    # Run examples (uncomment the one you want to test)
    
    # Example 1: Existing pose data
    # results = example_with_existing_pose_data()
    
    # Example 2: Video input (requires GPU and DeepLabCut)
    # results = example_with_video()
    
    # Example 3: Multiple extractors
    # results = example_multiple_extractors()
    
    # Example 4: Custom parameters
    # results = example_custom_parameters()
    
    print("See example_usage.py for usage examples")

