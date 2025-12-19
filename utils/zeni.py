import numpy as np
import pandas as pd


def zeni_algorithm(pose_data_with_rel, 
                   window_size=5,
                   min_contact_duration=3,
                   velocity_threshold=10,
                   error_mask=None):
    """
    Zeni algorithm for detecting foot strikes (paw-ground contacts).
    
    The algorithm detects foot strikes by finding:
    1. Local maxima in vertical position (y-coordinate) - paw at lowest point
    2. Low velocity at contact point
    3. Minimum contact duration to filter noise

    **CORE ASSUMPTIONS**:
    - vertical position indicates ground contact. may not hold if:
        - uneven terrain
        - camera is not orthogonal to subject
        - animal is climbing or jumping
    - smooth, continuous movements
    - animal is in stride/gait. may not hold if
        - complex gaits, animal is stationary
    - low velocity at contact. may not hold if
        - fast running
        - slipping
    - reference point stability
    - contact lasts for some minimum duration (multiple consecutive frames)
        - framerate must be consistent

    In a nutshell, we can tell if this algo is going to work based on our relative x and y position graphs. If they look terrible, its not gonna work
    
    Parameters:
    -----------
    pose_data_with_rel : pd.DataFrame
        DataFrame with relative positions (from pose_data_with_rel)
    window_size : int, default=5
        Window size for finding local maxima (should be odd)
    min_contact_duration : int, default=3
        Minimum number of consecutive frames for a valid contact
    velocity_threshold : float, default=10
        Maximum velocity (pixels/frame) at contact point
    error_mask : pd.DataFrame
        DataFrame with boolean columns indicating problematic frames for each bodypart.
        True values indicate frames with errors that should be filtered out.
        Must have columns matching paw names: 'front_left_paw', 'front_right_paw',
        'back_left_paw', 'back_right_paw'
    Returns:
    --------
    strikes : dict
        Dictionary with paw names as keys and lists of frame indices as values
        where foot strikes occur
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    
    scorer = pose_data_with_rel.columns.get_level_values(0).unique()[0]
    individual = pose_data_with_rel.columns.get_level_values(1).unique()[0]
    
    # Define paws and their relative bodypart names
    paws = {
        'front_left_paw': 'front_left_paw_rel',
        'front_right_paw': 'front_right_paw_rel',
        'back_left_paw': 'back_left_paw_rel',
        'back_right_paw': 'back_right_paw_rel'
    }
    
    strikes = {}
    
    for paw_name, rel_bodypart in paws.items():
        # Extract relative y-coordinate (vertical position)
        y_rel = pose_data_with_rel[(scorer, individual, rel_bodypart, 'y')]
        
        # Filter using error_mask
        if error_mask is None:
            raise ValueError("error_mask is required. Provide a DataFrame with boolean columns for each paw.")
        
        valid_mask = ~error_mask[paw_name]
        y_valid = y_rel.copy()
        y_valid[~valid_mask] = np.nan  # Mark invalid points
        
        # Fill NaN values: forward fill first, then backward fill for any remaining NaNs
        y_filled = y_valid.ffill().bfill()
        
        # Check if we still have NaN values (can happen if entire series is invalid)
        if y_filled.isna().all():
            # No valid data for this paw
            strikes[paw_name] = []
            continue
        
        # Convert to numpy array for uniform_filter1d
        y_array_filled = y_filled.values
        
        # Smooth the signal to reduce noise
        y_smooth = pd.Series(
            uniform_filter1d(y_array_filled, size=window_size),
            index=y_valid.index
        )
        
        # Calculate velocity (change in y per frame)
        velocity = np.abs(y_smooth.diff())
        
        # Find local maxima in y (lowest vertical position = foot strike)
        # Use prominence to avoid detecting small fluctuations
        # Prominence = minimum height difference from surrounding peaks
        y_array = y_smooth.values
        valid_indices = ~np.isnan(y_array)
        
        if np.sum(valid_indices) < window_size:
            # Not enough valid data
            strikes[paw_name] = []
            continue
        
        # Calculate prominence threshold (e.g., 5% of signal range)
        y_range = np.nanmax(y_array) - np.nanmin(y_array)
        prominence = max(y_range * 0.05, 5)  # At least 5 pixels or 5% of range
        
        # Find peaks (local maxima)
        peaks, properties = find_peaks(
            y_array,
            prominence=prominence,
            distance=window_size  # Minimum distance between peaks
        )
        
        # Filter peaks by velocity and error_mask
        valid_strikes = []
        for peak_idx in peaks:
            frame_idx = y_smooth.index[peak_idx]
            
            # Check velocity at contact point (should be low)
            if peak_idx < len(velocity):
                vel_at_contact = velocity.iloc[peak_idx]
                # Skip if velocity is NaN (first frame) or exceeds threshold
                if pd.isna(vel_at_contact) or vel_at_contact > velocity_threshold:
                    continue  # Too much movement, not a stable contact
            
            # Check error_mask at peak
            if error_mask[paw_name].iloc[peak_idx]:
                continue  # Frame has error, skip
            
            # Check if this is part of a contact duration
            # Look for consecutive frames around the peak with similar y values
            contact_window = window_size
            start_idx = max(0, peak_idx - contact_window // 2)
            end_idx = min(len(y_smooth), peak_idx + contact_window // 2 + 1)
            
            window_y = y_smooth.iloc[start_idx:end_idx]
            window_error_mask = error_mask[paw_name].iloc[start_idx:end_idx]
            
            # Count valid frames in window (no errors and not NaN)
            valid_in_window = np.sum(
                (~window_error_mask) & 
                (~np.isnan(window_y))
            )
            
            if valid_in_window >= min_contact_duration:
                valid_strikes.append(int(frame_idx))
        
        strikes[paw_name] = sorted(valid_strikes)
    
    return strikes