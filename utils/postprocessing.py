import numpy as np
import pandas as pd

# =============== ERROR DETECTION FUNCTIONS =============== #

def detect_velocity_errors(df, bodyparts, threshold_pixels=50, window_size=5, 
                          scorer=None, individual=None):
    """
    Vectorized detection of frames where keypoints jump too far (high velocity).
    Uses hybrid approach: extract with pandas, compute with numpy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns)
    bodyparts : list
        List of bodypart names to check
    threshold_pixels : float, default=50
        Maximum expected pixel movement per frame
    window_size : int, default=5
        Window size for median filter (smooth out noise)
    scorer : str, optional
        Scorer name (extracted from df if not provided)
    individual : str, optional
        Individual name (extracted from df if not provided)
    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with one column per bodypart (boolean), indicating errors
    """
    if scorer is None:
        scorer = df.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = df.columns.get_level_values(1).unique()[0]
    
    # Extract all data efficiently using pandas MultiIndex slicing
    x_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('x', level='coords', axis=1)[bodyparts]
    y_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('y', level='coords', axis=1)[bodyparts]
    likelihood_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('likelihood', level='coords', axis=1)[bodyparts]
    
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
    
    # Return DataFrame with one column per bodypart
    error_df = pd.DataFrame(errors, index=df.index, columns=bodyparts)
    
    return error_df


def detect_likelihood_errors(df, bodyparts, min_likelihood=0.5, 
                            scorer=None, individual=None):
    """
    Vectorized detection of frames with low likelihood scores.
    Uses hybrid approach: extract with pandas, compute with numpy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns)
    bodyparts : list
        List of bodypart names to check
    min_likelihood : float, default=0.5
        Minimum acceptable likelihood score
    scorer : str, optional
        Scorer name (extracted from df if not provided)
    individual : str, optional
        Individual name (extracted from df if not provided)
    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with one column per bodypart (boolean), indicating errors
    """
    if scorer is None:
        scorer = df.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = df.columns.get_level_values(1).unique()[0]
    
    # Extract all likelihood data efficiently
    likelihood_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('likelihood', level='coords', axis=1)[bodyparts]
    
    # Convert to numpy for fast computation
    likelihood_arr = likelihood_data.values
    
    # Vectorized error detection: likelihood < min_likelihood
    errors = likelihood_arr < min_likelihood
    
    # Return DataFrame with one column per bodypart
    error_df = pd.DataFrame(errors, index=df.index, columns=bodyparts)
    
    return error_df


def detect_distance_errors(df, bodyparts, reference_map, 
                           max_distance=300, scorer=None, individual=None):
    """
    Vectorized detection of frames where keypoints are too far from a reference point.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data (MultiIndex columns)
    bodyparts : list
        List of bodypart names to check (all bodyparts that will be in the output)
    reference_map : dict {str: list}
        Mapping of reference points to their respective bodyparts. 
        Example: {'back_base': ['front_left_paw', 'front_right_paw'], 
                  'tail_base': ['back_left_paw', 'back_right_paw']}
    max_distance : float, default=300
        Maximum expected distance in pixels
    scorer : str, optional
        Scorer name (extracted from df if not provided)
    individual : str, optional
        Individual name (extracted from df if not provided)
    
    Returns:
    --------
    error_df : pd.DataFrame
        DataFrame with one column per bodypart (boolean), indicating errors
    """
    if scorer is None:
        scorer = df.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = df.columns.get_level_values(1).unique()[0]
    
    # Initialize result DataFrame with all bodyparts
    error_df = pd.DataFrame(False, index=df.index, columns=bodyparts)
    
    # Process each reference point and its associated bodyparts
    for ref_point, ref_bodyparts in reference_map.items():
        # Extract reference point coordinates
        ref_x = df[(scorer, individual, ref_point, 'x')].values  # Shape: (n_frames,)
        ref_y = df[(scorer, individual, ref_point, 'y')].values  # Shape: (n_frames,)
        
        # Extract all bodypart data for this reference point vectorized
        x_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('x', level='coords', axis=1)[ref_bodyparts]
        y_data = df.xs((scorer, individual), level=[0, 1], axis=1).xs('y', level='coords', axis=1)[ref_bodyparts]
        
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
        
        # Store results in the output DataFrame
        error_df[ref_bodyparts] = errors
    
    return error_df

def detect_pose_errors(df, bodyparts=None, 
                       velocity_threshold=50,
                       min_likelihood=0.5,
                       max_distance=300,
                       reference_map=None,
                       scorer=None,
                       individual=None):
    """
    Comprehensive error detection combining multiple methods.
    Now returns per-bodypart error information.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pose data
    bodyparts : list, optional
        List of bodypart names (None = use all)
    velocity_threshold : float, default=50
        Max pixels per frame
    min_likelihood : float, default=0.5
        Minimum acceptable likelihood
    max_distance : float, default=300
        Max distance from reference point
    reference_map : dict, optional
        Mapping of reference points to their respective bodyparts.
        Example: {'back_base': ['front_left_paw', 'front_right_paw'], 
                  'tail_base': ['back_left_paw', 'back_right_paw']}
        If None, will create a default mapping using 'back_base' for all bodyparts
    scorer : str, optional
        Scorer name (extracted from df if not provided)
    individual : str, optional
        Individual name (extracted from df if not provided)
    
    Returns:
    --------
    error_details : pd.DataFrame
        DataFrame with error information:
        - Per-bodypart columns for each error type (e.g., 'front_left_paw_velocity')
    """
    if bodyparts is None:
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
    
    # Create default reference_map if not provided
    if reference_map is None:
        # Default: use 'back_base' as reference for all bodyparts
        reference_map = {
            'back_base': ['front_left_paw', 'front_right_paw'], 
            'tail_base': ['back_left_paw', 'back_right_paw']
            }
    
    # Run all detection methods (now return DataFrames with per-bodypart columns)
    velocity_errors_df = detect_velocity_errors(
        df, bodyparts, velocity_threshold, scorer=scorer, individual=individual
    )
    likelihood_errors_df = detect_likelihood_errors(
        df, bodyparts, min_likelihood, scorer=scorer, individual=individual
    )
    distance_errors_df = detect_distance_errors(
        df, bodyparts, reference_map, max_distance, 
        scorer=scorer, individual=individual
    )
    
    # Create error_details DataFrame with per-bodypart information
    error_details = pd.DataFrame(index=df.index)
    
    # Add per-bodypart error columns
    for bodypart in bodyparts:
        error_details[f'{bodypart}_velocity'] = velocity_errors_df[bodypart]
        error_details[f'{bodypart}_likelihood'] = likelihood_errors_df[bodypart]
        error_details[f'{bodypart}_distance'] = distance_errors_df[bodypart]
        error_details[f'{bodypart}_error'] = (velocity_errors_df[bodypart] | likelihood_errors_df[bodypart] | distance_errors_df[bodypart])
    
    return error_details


# =============== NORMALIZATION FUNCTIONS =============== #

def get_coords(df, bodypart, coord, scorer, individual):
    """Extract coordinates for a specific body part."""
    coords = df[(scorer, individual, bodypart)][[coord]]

    return coords

def paw_to_relative_position(df, scorer=None, individual=None, append_to_df=True):
    """
    Convert paw coordinates to relative position coordinates.
    """
    # Create new dataframe with same structure as original, including relative positions as new bodyparts
    # Start with a copy of the original data
    df_with_rel = df.copy()

    if scorer is None:
        scorer = df.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = df.columns.get_level_values(1).unique()[0]

    front_paws = ['front_left_paw', 'front_right_paw']
    back_paws = ['back_left_paw', 'back_right_paw']
    front_ref = 'back_base'
    back_ref = 'tail_base'

    back_base_x = get_coords(df, front_ref, 'x', scorer, individual)
    back_base_y = get_coords(df, front_ref, 'y', scorer, individual)
    tail_base_x = get_coords(df, back_ref, 'x', scorer, individual)
    tail_base_y = get_coords(df, back_ref, 'y', scorer, individual)

    # Calculate relative positions for front paws (relative to back_base)
    for paw in front_paws:
        paw_x = get_coords(df, paw, 'x', scorer, individual)
        paw_y = get_coords(df, paw, 'y', scorer, individual)
        
        # Create new bodypart name for relative position
        rel_bodypart = f'{paw}_rel'
        
        # Add relative x and y coordinates with same MultiIndex structure
        df_with_rel[(scorer, individual, rel_bodypart, 'x')] = paw_x - back_base_x
        df_with_rel[(scorer, individual, rel_bodypart, 'y')] = paw_y - back_base_y
        # Set likelihood to same as original paw
        df_with_rel[(scorer, individual, rel_bodypart, 'likelihood')] = df_with_rel[(scorer, individual, paw, 'likelihood')]

    # Calculate relative positions for back paws (relative to tail_base)
    for paw in back_paws:
        paw_x = get_coords(df, paw, 'x', scorer, individual)
        paw_y = get_coords(df, paw, 'y', scorer, individual)
        
        # Create new bodypart name for relative position
        rel_bodypart = f'{paw}_rel'
        
        # Add relative x and y coordinates with same MultiIndex structure
        df_with_rel[(scorer, individual, rel_bodypart, 'x')] = paw_x - tail_base_x
        df_with_rel[(scorer, individual, rel_bodypart, 'y')] = paw_y - tail_base_y
        # Set likelihood to same as original paw
        df_with_rel[(scorer, individual, rel_bodypart, 'likelihood')] = df_with_rel[(scorer, individual, paw, 'likelihood')]

    # Sort columns lexicographically to maintain proper MultiIndex order
    df_with_rel = df_with_rel.sort_index(axis=1)

    print("Dataframe with relative positions created:")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_with_rel.shape}")
    print(f"\nNew bodyparts added:")
    new_bodyparts = [bp for bp in df_with_rel.columns.get_level_values('bodyparts').unique() 
                    if bp not in df.columns.get_level_values('bodyparts').unique()]
    print(new_bodyparts)
    
    if append_to_df:
        return df_with_rel
    else:
        # Extract only the relative position columns
        # Need to construct full MultiIndex tuples: (scorer, individual, bodypart, coord)
        cols = []
        for bodypart in new_bodyparts:
            for coord in ['x', 'y', 'likelihood']:
                cols.append((scorer, individual, bodypart, coord))
        return df_with_rel[cols]