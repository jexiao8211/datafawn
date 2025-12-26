"""
Utility functions for analyzing events and extracting speed information.
"""

import numpy as np
import math
from typing import Dict, Any


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

