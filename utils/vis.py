import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

# ========== BODYPART MOVEMENT VISUALIZATION ========== #
def plot_bodypart_movement(data, bodypart, individual='animal0', min_likelihood=0.0, 
                           figsize=(15, 5), show_trajectory=True):
    """
    Plot the movement of a single body part over time.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DeepLabCut output DataFrame
    bodypart : str
        Name of the body part to plot (e.g., 'nose', 'tail_end')
    individual : str, default='animal0'
        Which individual to plot
    min_likelihood : float, default=0.0
        Minimum likelihood threshold (filter out low-confidence detections)
    figsize : tuple, default=(15, 5)
        Figure size for the plots
    show_trajectory : bool, default=True
        Whether to show the 2D trajectory plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Extract coordinates for the specified body part
    try:
        coords = data.xs((individual, bodypart, 'superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_'), level=['individuals', 'bodyparts', 'scorer'], axis=1)
    except KeyError:
        raise ValueError(f"Body part '{bodypart}' or individual '{individual}' not found in data")
    
    # Extract x, y, and likelihood
    x = coords['x'].values
    y = coords['y'].values
    likelihood = coords['likelihood'].values
    frames = coords.index.values
    
    # Filter by likelihood
    valid_mask = likelihood >= min_likelihood
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    frames_valid = frames[valid_mask]
    
    # Create figure with subplots
    if show_trajectory:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = list(axes) + [None]  # Pad for indexing
    
    # Plot 1: X position over time
    axes[0].plot(frames_valid, x_valid, 'b-', linewidth=1.5, alpha=0.7, label='x position')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].set_title(f'{bodypart} - X Position Over Time\n({individual})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Y position over time
    axes[1].plot(frames_valid, y_valid, 'r-', linewidth=1.5, alpha=0.7, label='y position')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].set_title(f'{bodypart} - Y Position Over Time\n({individual})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: 2D Trajectory (if enabled)
    if show_trajectory:
        # Color by frame number for trajectory
        scatter = axes[2].scatter(x_valid, y_valid, c=frames_valid, cmap='viridis', 
                                  s=10, alpha=0.6, edgecolors='none')
        axes[2].plot(x_valid, y_valid, 'k-', linewidth=0.5, alpha=0.3)
        axes[2].set_xlabel('X Position (pixels)')
        axes[2].set_ylabel('Y Position (pixels)')
        axes[2].set_title(f'{bodypart} - 2D Trajectory\n({individual})')
        axes[2].grid(True, alpha=0.3)
        axes[2].invert_yaxis()  # Invert y-axis to match image coordinates
        plt.colorbar(scatter, ax=axes[2], label='Frame')
    
    plt.tight_layout()
    return fig



def plot_bodypart_movement_enhanced(data, bodypart, individual='animal0', 
                                    min_likelihood=0.0, figsize=(18, 6)):
    """
    Enhanced version with velocity and statistics.
    """
    # Extract coordinates
    coords = data.xs((individual, bodypart, 'superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_'), level=['individuals', 'bodyparts', 'scorer'], axis=1)
    x = coords['x'].values
    y = coords['y'].values
    likelihood = coords['likelihood'].values
    frames = coords.index.values
    
    # Filter by likelihood
    valid_mask = likelihood >= min_likelihood
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    frames_valid = frames[valid_mask]
    
    # Calculate velocity (pixels per frame)
    if len(x_valid) > 1:
        dx = np.diff(x_valid)
        dy = np.diff(y_valid)
        velocity = np.sqrt(dx**2 + dy**2)
        frames_vel = frames_valid[:-1]
    else:
        velocity = np.array([])
        frames_vel = np.array([])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: X position
    axes[0, 0].plot(frames_valid, x_valid, 'b-', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('X Position (pixels)')
    axes[0, 0].set_title(f'{bodypart} - X Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y position
    axes[0, 1].plot(frames_valid, y_valid, 'r-', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Y Position (pixels)')
    axes[0, 1].set_title(f'{bodypart} - Y Position')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 2D Trajectory
    scatter = axes[1, 0].scatter(x_valid, y_valid, c=frames_valid, cmap='viridis', 
                                 s=10, alpha=0.6)
    axes[1, 0].plot(x_valid, y_valid, 'k-', linewidth=0.5, alpha=0.3)
    axes[1, 0].set_xlabel('X Position (pixels)')
    axes[1, 0].set_ylabel('Y Position (pixels)')
    axes[1, 0].set_title(f'{bodypart} - 2D Trajectory')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()
    plt.colorbar(scatter, ax=axes[1, 0], label='Frame')
    
    # Plot 4: Velocity
    if len(velocity) > 0:
        axes[1, 1].plot(frames_vel, velocity, 'g-', linewidth=1.5, alpha=0.7)
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Velocity (pixels/frame)')
        axes[1, 1].set_title(f'{bodypart} - Velocity')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{bodypart} Movement Analysis ({individual})', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# ========== VIDEO EXTRACTION ========== #
def get_frame_from_video(video_path, frame_number, display=False, convert_rgb=False):
    """
    Extract a specific frame from a video file.
    
    Parameters:
    -----------
    video_path : str or Path
        Path to the video file
    frame_number : int
        Frame number to extract (0-indexed)
    display : bool, default=False
        If True, display the frame using matplotlib
    convert_rgb : bool, default=False
        If True, convert BGR to RGB (useful for matplotlib display)
    
    Returns:
    --------
    np.ndarray or None
        Image array, or None if frame not found
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        raise ValueError(f"Frame number {frame_number} out of range. Video has {total_frames} frames (0-{total_frames-1})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB if requested (for matplotlib)
    if convert_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display if requested
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(frame if convert_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_number}')
        plt.axis('off')
        plt.show()
    
    return frame

# Usage:
# frame = get_frame_from_video('videos/vid1.mp4', 100, display=True, convert_rgb=True)


def extract_video_clip(video_path, start_frame, end_frame=None, output_path=None, 
                       fps=None, display=False):
    """
    Extract a range of frames from a video and save as a new video file.
    
    Parameters:
    -----------
    video_path : str or Path
        Path to the input video file
    start_frame : int
        Starting frame number (0-indexed)
    end_frame : int or None, default=None
        Ending frame number (0-indexed). If None, extracts to the end of video
    output_path : str or Path or None, default=None
        Path to save the output video. If None, auto-generates name
    fps : float or None, default=None
        Frames per second for output video. If None, uses original video FPS
    display : bool, default=False
        If True, display the first frame of the clip
    
    Returns:
    --------
    str
        Path to the saved output video file
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set end_frame if not provided
    if end_frame is None:
        end_frame = total_frames - 1
    
    # Validate frame range
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range. Video has {total_frames} frames")
    
    if end_frame < start_frame or end_frame >= total_frames:
        cap.release()
        raise ValueError(f"End frame {end_frame} out of range or before start frame")
    
    # Use provided fps or original
    output_fps = fps if fps is not None else original_fps
    
    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_frames_{start_frame}_to_{end_frame}.mp4"
    else:
        output_path = Path(output_path)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames
    frame_count = 0
    frames_to_extract = end_frame - start_frame + 1
    
    print(f"Extracting frames {start_frame} to {end_frame} ({frames_to_extract} frames)...")
    
    while frame_count < frames_to_extract:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        out.write(frame)
        
        # Display first frame if requested
        if display and frame_count == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(frame_rgb)
            plt.title(f'First frame of clip (Frame {start_frame})')
            plt.axis('off')
            plt.show()
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"Video clip saved to: {output_path}")
    return str(output_path)



# ========== ERROR DETECTION VISUALIZATION ========== #

def visualize_error_masks(error_details, bodyparts, figsize=(16, 12)):
    """
    Visualize error masks for each paw and each error type.
    
    Parameters:
    -----------
    error_details : pd.DataFrame
        Error details DataFrame from detect_pose_errors()
    bodyparts : list
        List of bodypart names (e.g., ['front_left_paw', 'front_right_paw', ...])
    figsize : tuple, default=(16, 12)
        Figure size
    """
    error_types = ['velocity', 'likelihood', 'distance', 'error']
    n_paws = len(bodyparts)
    n_types = len(error_types)
    
    # Create figure with subplots: one row per paw, one column per error type
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_paws, n_types, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color map for each error type
    colors = {
        'velocity': 'red',
        'likelihood': 'orange',
        'distance': 'purple',
        'error': 'black'
    }
    
    # Plot each paw and each error type
    for paw_idx, paw in enumerate(bodyparts):
        for type_idx, error_type in enumerate(error_types):
            ax = fig.add_subplot(gs[paw_idx, type_idx])
            
            # Get error mask for this paw and error type
            error_mask = error_details[f'{paw}_{error_type}']
            
            # Plot error mask
            ax.fill_between(
                error_mask.index,
                0,
                error_mask.astype(int),
                alpha=0.6,
                color=colors[error_type],
                label=f'{error_type} errors'
            )
            
            # Add statistics
            n_errors = error_mask.sum()
            error_pct = 100 * error_mask.mean()
            
            ax.set_title(f'{paw}\n{error_type.title()} Errors\n'
                        f'{n_errors} frames ({error_pct:.1f}%)',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('Error (1=error, 0=ok)', fontsize=9)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['OK', 'Error'])
    
    plt.suptitle('Error Masks by Paw and Error Type', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def visualize_error_masks_combined(error_details, bodyparts, figsize=(18, 10)):
    """
    Visualize all error types for each paw in a single plot per paw.
    Shows all three error types overlaid.
    
    Parameters:
    -----------
    error_details : pd.DataFrame
        Error details DataFrame from detect_pose_errors()
    bodyparts : list
        List of bodypart names
    figsize : tuple, default=(18, 10)
        Figure size
    """
    error_types = ['velocity', 'likelihood', 'distance', 'error']
    colors = {
        'velocity': 'red',
        'likelihood': 'orange',
        'distance': 'purple',
        'error': 'black'
    }
    
    n_paws = len(bodyparts)
    fig, axes = plt.subplots(n_paws, 1, figsize=figsize, sharex=True)
    
    if n_paws == 1:
        axes = [axes]
    
    for paw_idx, paw in enumerate(bodyparts):
        ax = axes[paw_idx]
        
        # Plot each error type
        for error_type in error_types:
            error_mask = error_details[f'{paw}_{error_type}']
            
            # Offset each error type slightly for visibility
            offset = error_types.index(error_type) * 0.3
            
            ax.fill_between(
                error_mask.index,
                offset,
                offset + error_mask.astype(float),
                alpha=0.7,
                color=colors[error_type],
                label=f'{error_type.title()} ({error_mask.sum()} frames)'
            )
        
        ax.set_title(f'{paw} - All Error Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Type', fontsize=10)
        ax.set_ylim(-0.2, 1.5)
        ax.set_yticks([0.15, 0.45, 0.75, 1.2])
        ax.set_yticklabels(['Velocity', 'Likelihood', 'Distance', 'Any'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Frame', fontsize=11)
    plt.suptitle('Error Masks by Paw (All Types Combined)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_error_summary(error_details, bodyparts, figsize=(14, 8)):
    """
    Create summary visualizations: error counts and percentages.
    
    Parameters:
    -----------
    error_details : pd.DataFrame
        Error details DataFrame from detect_pose_errors()
    bodyparts : list
        List of bodypart names
    figsize : tuple, default=(14, 8)
        Figure size
    """
    error_types = ['velocity', 'likelihood', 'distance', 'error']
    
    # Calculate statistics
    stats = []
    for paw in bodyparts:
        for error_type in error_types:
            mask = error_details[f'{paw}_{error_type}']
            stats.append({
                'Paw': paw,
                'Error Type': error_type.title(),
                'Count': mask.sum(),
                'Percentage': 100 * mask.mean()
            })
    
    stats_df = pd.DataFrame(stats)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Error counts by paw and type
    pivot_counts = stats_df.pivot(index='Paw', columns='Error Type', values='Count')
    pivot_counts.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Error Counts by Paw and Type', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Paw', fontsize=11)
    ax1.set_ylabel('Number of Error Frames', fontsize=11)
    ax1.legend(title='Error Type', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Error percentages by paw and type
    pivot_pct = stats_df.pivot(index='Paw', columns='Error Type', values='Percentage')
    pivot_pct.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Error Percentages by Paw and Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Paw', fontsize=11)
    ax2.set_ylabel('Error Percentage (%)', fontsize=11)
    ax2.legend(title='Error Type', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Print summary table
    print("\nError Summary Statistics:")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    print("=" * 60)
    
    return fig, stats_df


def visualize_error_timeline(error_details, bodyparts, 
                              figsize=(16, 6)):
    """
    Visualize error timeline showing when errors occur across all paws.
    
    Parameters:
    -----------
    error_details : pd.DataFrame
        Error details DataFrame from detect_pose_errors()
    bodyparts : list
        List of bodypart names
    error_mask : pd.Series, optional
        Overall error mask from detect_pose_errors()
    figsize : tuple, default=(16, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a combined error timeline
    # Stack errors from different paws vertically
    y_positions = {}
    for idx, paw in enumerate(bodyparts):
        y_positions[paw] = idx
    
    error_types = ['velocity', 'likelihood', 'distance', 'error']
    colors = {'velocity': 'red', 'likelihood': 'orange', 'distance': 'purple', 'error': 'black'}
    
    # Plot errors for each paw
    for paw in bodyparts:
        y_base = y_positions[paw]
        
        for error_type in error_types:
            mask = error_details[f'{paw}_{error_type}']
            error_frames = mask.index[mask]
            
            if len(error_frames) > 0:
                y_offset = error_types.index(error_type) * 0.25
                ax.scatter(
                    error_frames,
                    [y_base + y_offset] * len(error_frames),
                    c=colors[error_type],
                    s=20,
                    alpha=0.6,
                    label=f'{paw} {error_type}' if paw == bodyparts[0] else ''
                )
    
    
    ax.set_yticks([y_positions[paw] for paw in bodyparts])
    ax.set_yticklabels(bodyparts)
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Paw', fontsize=11)
    ax.set_title('Error Timeline Across All Paws', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    return fig


# ========== ZENI ALGORITHM VISUALIZATION ========== #
def plot_strikes(pose_data_with_rel, strikes, paw_name, crop_from, crop_to):
    """Plot paw movement with detected strikes marked."""
    import matplotlib.pyplot as plt
    
    if crop_from is not None:
        pose_data_with_rel = pose_data_with_rel.iloc[crop_from:]
    if crop_to is not None:
        pose_data_with_rel = pose_data_with_rel.iloc[:crop_to]

    scorer = pose_data_with_rel.columns.get_level_values(0).unique()[0]
    individual = pose_data_with_rel.columns.get_level_values(1).unique()[0]
    rel_bodypart = f'{paw_name}_rel'
    
    y_rel = pose_data_with_rel[(scorer, individual, rel_bodypart, 'y')]
    likelihood = pose_data_with_rel[(scorer, individual, rel_bodypart, 'likelihood')]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot y position
    ax.plot(y_rel.index, y_rel.values, 'b-', alpha=0.5, label='Y position')
    
    # Mark strikes
    strike_frames = strikes[paw_name]
    if len(strike_frames) > 0:
        strike_frames = [frame for frame in strike_frames if frame >= crop_from and frame <= crop_to]
        strike_y = y_rel.loc[strike_frames]
        ax.scatter(strike_frames, strike_y.values, color='red', s=100, 
                  marker='v', label='Foot strikes', zorder=5)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Relative Y position (pixels)')
    ax.set_title(f'{paw_name} - Foot Strikes Detected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

