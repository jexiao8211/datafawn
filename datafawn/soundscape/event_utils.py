"""
Utility functions for analyzing events and extracting speed information.
"""

import numpy as np
import math
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt


def get_speed_absolute(results: dict, window: int = 30) -> np.ndarray:
    """
    Calculate absolute speed (footfall count per window) for each frame.
    
    Unlike the relative speed functions, this returns the raw footfall count
    in each rolling window, NOT normalized to the maximum. This allows for
    absolute speed thresholds that work consistently across different videos.
    
    Parameters
    ----------
    results : dict
        Output dictionary from Zeni algorithm. Must contain an 'events' key.
    window : int, default=30
        Size of the rolling window in frames.
    
    Returns
    -------
    numpy.ndarray
        Array where each element is the count of footfalls in the window
        centered on that frame. Values are raw counts (e.g., 0, 1, 2, 3...).
    """
    if 'events' not in results:
        raise ValueError("Please enter in a valid results_ dictionary")
    
    # Collect all footfall frames
    foot_falls = []
    for (scorer, individual), event_dict in results['events'].items():
        for foot, frames in event_dict.items():
            for frame in frames:
                foot_falls.append(frame)
    
    if len(foot_falls) == 0:
        return np.zeros(1)
    
    foot_falls.sort()
    max_frame = foot_falls[-1]
    
    # Create binary array marking footfall frames
    footfall_binary = np.zeros(max_frame + 1, dtype=float)
    for f in foot_falls:
        footfall_binary[f] = 1.0
    
    # Create rolling window kernel (uniform weights)
    kernel = np.ones(window)
    
    # Apply rolling sum using convolution (centered window)
    rolling_count = np.convolve(footfall_binary, kernel, mode='same')
    
    return rolling_count


def plot_speed_array(
    speed_array: np.ndarray,
    fps: float = 30.0,
    title: str = "Speed Over Time",
    save_path: Optional[str] = None,
    threshold: Optional[float] = None
):
    """
    Plot the speed array as a line graph over time.
    
    Parameters
    ----------
    speed_array : numpy.ndarray
        Array of speed values (one per frame)
    fps : float, default=30.0
        Frames per second (to convert frames to seconds on x-axis)
    title : str, default="Speed Over Time"
        Title for the plot
    save_path : str, optional
        If provided, saves the plot to this path instead of displaying
    threshold : float, optional
        If provided, draws a horizontal line at this threshold value
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object (useful for further customization)
    """
    
    # Convert frames to seconds
    time_seconds = np.arange(len(speed_array)) / fps
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_seconds, speed_array, linewidth=0.8, color='steelblue')
    ax.fill_between(time_seconds, speed_array, alpha=0.3, color='steelblue')
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Threshold: {threshold}')
        ax.legend()
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speed (footfalls per window)')
    ax.set_title(title)
    ax.set_xlim(0, time_seconds[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    stats_text = f"Min: {np.min(speed_array):.1f}  Max: {np.max(speed_array):.1f}  Mean: {np.mean(speed_array):.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Speed plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def print_speed_stats(speed_array: np.ndarray, fps: float = 30.0):
    """
    Print statistics about the speed array to help choose thresholds.
    
    Parameters
    ----------
    speed_array : numpy.ndarray
        Array of speed values
    fps : float, default=30.0
        Frames per second
    """
    duration = len(speed_array) / fps
    
    print(f"\n{'='*50}")
    print("SPEED ARRAY STATISTICS")
    print(f"{'='*50}")
    print(f"Duration: {duration:.1f} seconds ({len(speed_array)} frames)")
    print(f"Min:  {np.min(speed_array):.2f} footfalls/window")
    print(f"Max:  {np.max(speed_array):.2f} footfalls/window")
    print(f"Mean: {np.mean(speed_array):.2f} footfalls/window")
    print(f"Std:  {np.std(speed_array):.2f}")
    
    # Percentiles for threshold guidance
    percentiles = [25, 50, 75, 90, 95]
    print(f"\nPercentiles (for threshold guidance):")
    for p in percentiles:
        val = np.percentile(speed_array, p)
        print(f"  {p}th percentile: {val:.2f}")
    
    # Time above various thresholds
    print(f"\nTime above threshold:")
    test_thresholds = [2, 4, 6, 8, 10]
    for t in test_thresholds:
        pct_above = (np.sum(speed_array >= t) / len(speed_array)) * 100
        print(f"  >= {t}: {pct_above:.1f}% of video")
    print(f"{'='*50}\n")


def get_speed_from_zeni_smooth(results: dict = {}, window: int = 30):
    """
    Calculate smooth relative running speed for each frame using a rolling window.
    
    Unlike get_speed_from_zeni, this function provides smooth, continuous speed
    values by using a rolling window centered on each frame. Every frame gets a
    speed value based on the footfall density in its surrounding window.
    
    Algorithm:
    1. Extract all footfall frame numbers from all individuals and all feet
    2. Create a binary array marking footfall frames
    3. Apply a rolling sum (convolution) to count footfalls in each window
    4. Normalize by the maximum count to get values between 0 and 1
    
    Parameters
    ----------
    results : dict
        Output dictionary from Zeni algorithm. Must contain an 'events' key with
        structure: {(scorer, individual): {foot_name: [frame_numbers]}}
    window : int, default=30
        Size of the rolling window in frames. Larger windows produce smoother
        speed curves but less temporal precision.
    
    Returns
    -------
    numpy.ndarray
        Array of shape (max_frame + 1,) where each element represents the
        normalized speed (0.0 to 1.0) for that frame. Values transition smoothly
        between frames.
    """
    if 'events' not in results:
        raise ValueError("Please enter in a valid results_ dictionary")
    
    # Collect all footfall frames
    foot_falls = []
    for (scorer, individual), event_dict in results['events'].items():
        for foot, frames in event_dict.items():
            for frame in frames:
                foot_falls.append(frame)
    
    if len(foot_falls) == 0:
        return np.zeros(1)
    
    foot_falls.sort()
    max_frame = foot_falls[-1]
    
    # Create binary array marking footfall frames
    footfall_binary = np.zeros(max_frame + 1, dtype=float)
    for f in foot_falls:
        footfall_binary[f] = 1.0
    
    # Create rolling window kernel (uniform weights)
    kernel = np.ones(window)
    
    # Apply rolling sum using convolution (centered window)
    # 'same' mode keeps output same length as input
    rolling_count = np.convolve(footfall_binary, kernel, mode='same')
    
    # Normalize to [0, 1] range
    max_count = np.max(rolling_count)
    if max_count > 0:
        speed_array = rolling_count / max_count
    else:
        speed_array = rolling_count
    
    return speed_array


def get_speed_from_zeni(results: dict = {}, window: int = 30):
    """
    Calculate relative running speed for each frame based on footfall frequency.
    
    This function analyzes footfall events from the Zeni algorithm to estimate the
    animal's running speed at each frame. Speed is determined by the frequency of
    footfalls within sliding windows: more footfalls per window indicates faster
    movement.
    
    Algorithm:
    1. Extract all footfall frame numbers from all individuals and all feet
    2. Create an array covering all frames from 0 to the last footfall
    3. Divide the video into non-overlapping windows of `window` frames each
    4. Count footfalls within each window to measure local activity/speed
    5. Find the window with the maximum footfall count (peak speed)
    6. Normalize each frame's speed by dividing its window's footfall count by
       the maximum count, resulting in values between 0 and 1
    
    The returned array maps each frame to a normalized speed value:
    - 0.0: No footfalls in that frame's window (slowest/stationary)
    - 1.0: Frame is in the window with maximum footfalls (fastest)
    - Values between 0 and 1: Proportional speed relative to peak speed
    
    Note: Frames without footfalls in their window are set to 0. Only frames
    that contain footfalls (or are in windows containing footfalls) get non-zero
    speed values.

    Parameters
    ------------
    results : dict
        Output dictionary from Zeni algorithm. Must contain an 'events' key with
        structure: {(scorer, individual): {foot_name: [frame_numbers]}}
        Example: {('scorer1', 'animal0'): {'front_left_paw_strike': [10, 50, 90]}}

    window : int, default=30
        Number of frames per analysis window. Larger windows provide smoother
        speed estimates but less temporal resolution. Smaller windows are more
        responsive to speed changes but may be noisier.

    Returns
    -------
    numpy.ndarray
        Array of shape (max_frame + 1,) where each element represents the
        normalized speed (0.0 to 1.0) for that frame number. Frames without
        footfalls in their window have value 0.0.

    Raises
    ------
    ValueError
        If 'events' key is missing from results dictionary.
    """
    if 'events' not in results:
        raise ValueError("Please enter in a valid results_ dictionary")
    
    foot_falls = []
    for (scorer, individual), event_dict in results['events'].items():
        for foot, frames in event_dict.items():
            for frame in frames:
                foot_falls.append(frame)
    
    foot_falls.sort()
    all_frames = np.zeros(shape=(foot_falls[-1] + 1), dtype=float)

    num_windows = math.ceil(len(all_frames) / window)

    highest_foot_falls_per_window = 0
    for i in range(num_windows):
        count = sum((i * window) <= x <= (i * window + window) for x in foot_falls)
        if count > highest_foot_falls_per_window:
            highest_foot_falls_per_window = count

    for i in range(num_windows):
        count = sum((i * window) <= x <= (i * window + window) for x in foot_falls)
        for f in foot_falls:
            if (i * window) <= f and f <= (i * window + window):
                all_frames[f] = count / highest_foot_falls_per_window
    
    # all_frames[frame] = 0 if there is no foot fall at that frame
    # all_frames[frame] != 0 if there is a foot fall at that frame
    # should be a value between 0 and 1
    # 1 meaning that the animal is moving at it's fastest
    # 0 meaning the animal is moving at it's slowest
    return all_frames

