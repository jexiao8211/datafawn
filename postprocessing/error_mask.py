import numpy as np
import pandas as pd


def detect_velocity_errors(df, bodyparts, threshold_pixels=50, window_size=5):
    """
    Vectorized detection of frames where keypoints jump too far (high velocity).
    Processes all (scorer, individual) combinations in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns: scorer, individual, bodypart, coords)
    bodyparts : list
        List of bodypart names to check
    threshold_pixels : float, default=50
        Maximum expected pixel movement per frame
    window_size : int, default=5
        Window size for median filter (smooth out noise)

    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with MultiIndex columns (scorer, individual, bodypart) containing boolean error flags
    """
    # Get all unique (scorer, individual) combinations
    scorers = df.columns.get_level_values(0).unique()
    individuals = df.columns.get_level_values(1).unique()

    
    # Build MultiIndex for output columns
    error_columns = pd.MultiIndex.from_product(
        [scorers, individuals, bodyparts],
        names=['scorer', 'individual', 'bodypart']
    )
    error_df = pd.DataFrame(False, index=df.index, columns=error_columns)
    
    # Process each (scorer, individual) combination
    for scorer_name in scorers:
        for individual_name in individuals:
            try:
                # Extract all data efficiently using pandas MultiIndex slicing
                x_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('x', level='coords', axis=1)[bodyparts]
                y_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('y', level='coords', axis=1)[bodyparts]
                likelihood_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('likelihood', level='coords', axis=1)[bodyparts]
                
                # Convert to numpy for fast computation
                x_arr = x_data.values  # Shape: (n_frames, n_bodyparts)
                y_arr = y_data.values
                likelihood_arr = likelihood_data.values
                
                # Vectorized velocity calculation
                dx = np.abs(np.diff(x_arr, axis=0, prepend=x_arr[0:1]))
                dy = np.abs(np.diff(y_arr, axis=0, prepend=y_arr[0:1]))
                velocity = np.sqrt(dx**2 + dy**2)
                
                # Smooth using pandas rolling (convenient for this)
                velocity_df = pd.DataFrame(velocity, index=df.index, columns=bodyparts)
                velocity_smooth = velocity_df.rolling(window=window_size, center=True).median()
                
                # Vectorized error detection: (velocity > threshold) & (likelihood > 0.1)
                errors = (velocity_smooth.values > threshold_pixels) & (likelihood_arr > 0.1)
                
                # Store results in the output DataFrame
                for i, bodypart in enumerate(bodyparts):
                    error_df[(scorer_name, individual_name, bodypart)] = errors[:, i]
            except (KeyError, IndexError):
                # Skip if this (scorer, individual) combination doesn't exist
                continue
    
    return error_df


def detect_likelihood_errors(df, bodyparts, min_likelihood=0.5):
    """
    Vectorized detection of frames with low likelihood scores.
    Processes all (scorer, individual) combinations in the DataFrame.
    Uses hybrid approach: extract with pandas, compute with numpy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns: scorer, individual, bodypart, coords)
    bodyparts : list
        List of bodypart names to check
    min_likelihood : float, default=0.5
        Minimum acceptable likelihood score

    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with MultiIndex columns (scorer, individual, bodypart) containing boolean error flags
    """
    # Get all unique (scorer, individual) combinations
    scorers = df.columns.get_level_values(0).unique()
    individuals = df.columns.get_level_values(1).unique()
    
    
    # Build MultiIndex for output columns
    error_columns = pd.MultiIndex.from_product(
        [scorers, individuals, bodyparts],
        names=['scorer', 'individual', 'bodypart']
    )
    error_df = pd.DataFrame(False, index=df.index, columns=error_columns)
    
    # Process each (scorer, individual) combination
    for scorer_name in scorers:
        for individual_name in individuals:
            try:
                # Extract all likelihood data efficiently
                likelihood_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('likelihood', level='coords', axis=1)[bodyparts]
                
                # Convert to numpy for fast computation
                likelihood_arr = likelihood_data.values
                
                # Vectorized error detection: likelihood < min_likelihood
                errors = likelihood_arr < min_likelihood
                
                # Store results in the output DataFrame
                for i, bodypart in enumerate(bodyparts):
                    error_df[(scorer_name, individual_name, bodypart)] = errors[:, i]
            except (KeyError, IndexError):
                # Skip if this (scorer, individual) combination doesn't exist
                continue
    
    return error_df


def detect_distance_errors(df, bodyparts, reference_map, 
                           max_distance=300):
    """
    Vectorized detection of frames where keypoints are too far from a reference point.
    Processes all (scorer, individual) combinations in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns: scorer, individual, bodypart, coords)
    bodyparts : list
        List of bodypart names to check (all bodyparts that will be in the output)
    reference_map : dict {str: list}
        Mapping of reference points to their respective bodyparts. 
        Example: {'back_base': ['front_left_paw', 'front_right_paw'], 
                  'tail_base': ['back_left_paw', 'back_right_paw']}
    max_distance : float, default=300
        Maximum expected distance in pixels

    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with MultiIndex columns (scorer, individual, bodypart) containing boolean error flags
    """
    # Get all unique (scorer, individual) combinations
    scorers = df.columns.get_level_values(0).unique()
    individuals = df.columns.get_level_values(1).unique()

    
    # Build MultiIndex for output columns
    error_columns = pd.MultiIndex.from_product(
        [scorers, individuals, bodyparts],
        names=['scorer', 'individual', 'bodypart']
    )
    error_df = pd.DataFrame(False, index=df.index, columns=error_columns)
    
    # Process each (scorer, individual) combination
    for scorer_name in scorers:
        for individual_name in individuals:
            try:
                # Initialize temporary DataFrame for this (scorer, individual) combination
                temp_error_df = pd.DataFrame(False, index=df.index, columns=bodyparts)
                
                # Process each reference point and its associated bodyparts
                for ref_point, ref_bodyparts in reference_map.items():
                    # Extract reference point coordinates
                    ref_x = df[(scorer_name, individual_name, ref_point, 'x')].values  # Shape: (n_frames,)
                    ref_y = df[(scorer_name, individual_name, ref_point, 'y')].values  # Shape: (n_frames,)
                    
                    # Extract all bodypart data for this reference point vectorized
                    x_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('x', level='coords', axis=1)[ref_bodyparts]
                    y_data = df.xs((scorer_name, individual_name), level=[0, 1], axis=1).xs('y', level='coords', axis=1)[ref_bodyparts]
                    
                    # Convert to numpy for fast computation
                    x_arr = x_data.values  # Shape: (n_frames, n_ref_bodyparts)
                    y_arr = y_data.values
                    
                    # Vectorized distance calculation
                    # Broadcast reference point to match bodyparts shape
                    ref_x_broadcast = ref_x[:, np.newaxis]  # Shape: (n_frames, 1)
                    ref_y_broadcast = ref_y[:, np.newaxis]  # Shape: (n_frames, 1)
                    
                    dx = x_arr - ref_x_broadcast
                    dy = y_arr - ref_y_broadcast
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Vectorized error detection: (distance > max_distance)
                    errors = (distance > max_distance)
                    
                    # Store results in the temporary DataFrame
                    temp_error_df[ref_bodyparts] = errors
                
                # Store results in the output DataFrame
                for bodypart in bodyparts:
                    error_df[(scorer_name, individual_name, bodypart)] = temp_error_df[bodypart]
            except (KeyError, IndexError):
                # Skip if this (scorer, individual) combination doesn't exist
                continue
    
    return error_df

