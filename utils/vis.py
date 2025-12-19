import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

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