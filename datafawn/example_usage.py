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


# =============== CONFIGURATION =============== #
DATA_PATH = "processed_vids/dog_running_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_.h5"
VIDEO_PATH = "LSTM/dog_running.mp4"
LSTM_MODEL_PATH = "models/trained_lstm.pt"  # Path to your trained LSTM model


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
# Zeni provides rule-based extraction (used for training labels)
zeni_extractor = ZeniExtractor()


# =============== CLASSIFIERS =============== #
# LSTM runs AFTER Zeni and provides ML-based classification
lstm_classifier = LSTMEventClassifier(
    model_path=LSTM_MODEL_PATH,  # Path to trained model
    input_dim=None,  # Will be inferred from data
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    threshold=0.5,
    device=device,
    sequence_length=30,  # Process in 30-frame sequences
    use_sliding_window=True,  # Use sliding window for better coverage
    stride=5  # 5-frame stride for overlapping predictions
)


# =============== CREATE 4-STAGE PIPELINE =============== #
print("\n=== Running 4-Stage Pipeline ===")
print("Stage 1: Pose Estimation")
print("Stage 2: Postprocessing")
print("Stage 3: Event Extraction (Zeni)")
print("Stage 4: Classification (LSTM)")
print()

pipeline = EventDetectionPipeline(
    pose_estimator=dlc_estimator,
    postprocessors=[error_pp, rel_pp],
    event_extractors=[zeni_extractor],  # Stage 3: Rule-based
    classifiers=[lstm_classifier]        # Stage 4: ML-based (NEW!)
)

# Run pipeline
try:
    results = pipeline.run(pose_data_path=DATA_PATH)
    
    print("\n=== Pipeline Results ===")
    print(f"Pose data shape: {results['pose_data'].shape}")
    print(f"Postprocessed data shape: {results['postprocessed_data'].shape}")
    print(f"Event extractors run: {results['metadata']['extractor_names']}")
    print(f"Classifiers run: {results['metadata']['classifier_names']}")
    
    # =============== STAGE 3: EVENT EXTRACTION RESULTS =============== #
    print("\n=== Stage 3: Event Extraction (Zeni) ===")
    zeni_results = results['events']['ZeniExtractor']
    print(f"Zeni events: {zeni_results}")
    
    # =============== STAGE 4: CLASSIFICATION RESULTS =============== #
    print("\n=== Stage 4: Classification (LSTM) ===")
    lstm_results = results['classifications']['LSTMEventClassifier']
    print(f"Number of strikes detected: {lstm_results['n_strikes']}")
    print(f"Strike frames: {lstm_results['strike_frames']}")
    print(f"Events (grouped): {lstm_results['events']}")
    print(f"Threshold used: {lstm_results['metadata']['threshold']}")
    
    # =============== COMPARISON =============== #
    print("\n=== Comparison: Zeni vs LSTM ===")
    
    # Extract Zeni strike frames (format may vary)
    zeni_strikes = []
    if isinstance(zeni_results, dict) and 'events' in zeni_results:
        zeni_strikes = zeni_results.get('events', [])
    elif isinstance(zeni_results, list):
        zeni_strikes = zeni_results
    
    print(f"Zeni detected: {len(zeni_strikes) if isinstance(zeni_strikes, list) else 'N/A'} events")
    print(f"LSTM detected: {lstm_results['n_strikes']} strike frames")
    
    # =============== VISUALIZE PREDICTIONS =============== #
    print("\n=== Frame-by-Frame Predictions (first 100 frames) ===")
    print("Format: Frame | LSTM Pred | LSTM Prob | Status")
    print("-" * 50)
    
    for i in range(min(100, len(lstm_results['frame_predictions']))):
        pred = lstm_results['frame_predictions'][i]
        prob = lstm_results['frame_probabilities'][i]
        
        status = ""
        if pred == 1:
            status = " ← STRIKE DETECTED"
        elif prob > 0.3:
            status = " (uncertain)"
        
        print(f"Frame {i:3d} | {pred} | {prob:.3f}{status}")
    
    # =============== SAVE RESULTS =============== #
    import json
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LSTM predictions
    lstm_output = output_dir / "lstm_predictions.json"
    with open(lstm_output, 'w') as f:
        json.dump(lstm_results, f, indent=2)
    print(f"\nLSTM predictions saved to {lstm_output}")
    
    # Save comparison
    comparison = {
        'zeni': zeni_results,
        'lstm': lstm_results,
        'metadata': results['metadata']
    }
    comparison_output = output_dir / "zeni_vs_lstm.json"
    with open(comparison_output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {comparison_output}")

except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("\n=== LSTM Model Not Found ===")
    print("The LSTM model needs to be trained first. Follow these steps:")
    print()
    print("1. Generate training data using Zeni labels:")
    print("   python create_lstm_training_data.py")
    print()
    print("2. Train the LSTM model:")
    print("   python train_lstm.py")
    print()
    print("3. Run this pipeline again with the trained model")
    print()
    print("For now, running pipeline with Zeni only...")
    print()
    
    # Fallback: run with just Zeni
    pipeline_zeni_only = EventDetectionPipeline(
        pose_estimator=dlc_estimator,
        postprocessors=[error_pp, rel_pp],
        event_extractors=[zeni_extractor],
        classifiers=[]  # No classifiers
    )
    
    results = pipeline_zeni_only.run(pose_data_path=DATA_PATH)
    print(f"Zeni results: {results['events']}")
    print("\nTo use LSTM classification, train the model first!")


# =============== EXAMPLE: GENERATE TRAINING DATA =============== #
print("\n\n=== Bonus: How to Generate Training Data ===")
print("Use Zeni results as labels for LSTM training:")
print()
print("```python")
print("# Run pipeline with only Zeni")
print("results = pipeline.run(pose_data_path=DATA_PATH)")
print()
print("# Extract features and labels")
print("features = results['postprocessed_data'].values")
print("zeni_events = results['events']['ZeniExtractor']")
print()
print("# Convert Zeni events to frame-level binary labels")
print("labels = create_binary_labels(zeni_events, n_frames=len(features))")
print()
print("# Save for training")
print("np.savez('training_data.npz', features=features, labels=labels)")
print("```")