import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def zeni_algorithm(pose_data_with_rel,
                   smooth_window_size=5,
                   prominence_percentage=0.05,
                   orientation_likelihood_threshold=0.0,
                   orientation_smooth_window_size=15,
                   show_plots=True):
    """
    Simplified Zeni algorithm for detecting foot strikes.
        
    Parameters:
    -----------
    pose_data_with_rel : pd.DataFrame
        DataFrame with relative positions (from pose_data_with_rel)
    smooth_window_size : int, default=5
        Window size for smoothing position signals
    prominence_percentage : float, default=0.05
        Minimum prominence for peak detection as percentage of x range
    orientation_likelihood_threshold : float, default=0.0
        Minimum likelihood for orientation data to be considered valid.
    orientation_smooth_window_size : int, default=15
        Window size for smoothing orientation when determining direction of motion.
        Orientation is computed from the difference between front (back_base) and back (tail_base) bodyparts.
        Larger values make direction changes less sensitive to noise.
    show_plots : bool, default=True
        If True, displays plots immediately. If False, returns figure objects without displaying
    
    Returns:
    --------
    strikes : dict
        Nested dictionary: strikes[(scorer, individual)][paw_name] contains
        a list of frame indices where foot strikes occur for that combination.
        Structure: {(scorer, individual): {paw_name: [frame_indices]}}
    """

    # Get all unique (scorer, individual) combinations
    scorers = pose_data_with_rel.columns.get_level_values(0).unique()
    individuals = pose_data_with_rel.columns.get_level_values(1).unique()

    # Define paws and their relative bodypart names
    rel_paws = {
        'front_left_paw_rel',
        'front_right_paw_rel',
        'back_left_paw_rel',
        'back_right_paw_rel'
    }

    front_bodypart = 'back_base'  # Front of the animal
    back_bodypart = 'tail_base'    # Back of the animal


    strikes = {}
    # Process each (scorer, individual) combination
    for scorer in scorers:
        for individual in individuals:

            # Initialize nested dictionaries for this combination
            if (scorer, individual) not in strikes:
                strikes[(scorer, individual)] = {}

            # GET ORIENTATION DATA
            # Use the difference between front (back_base) and back (tail_base) to determine orientation
            front_x = pose_data_with_rel[(scorer, individual, front_bodypart, 'x')]
            back_x = pose_data_with_rel[(scorer, individual, back_bodypart, 'x')]
            front_likelihood = pose_data_with_rel[(scorer, individual, front_bodypart, 'likelihood')]
            back_likelihood = pose_data_with_rel[(scorer, individual, back_bodypart, 'likelihood')]
            
            # Apply likelihood filter - mask low-likelihood values
            front_errors = front_likelihood < orientation_likelihood_threshold
            back_errors = back_likelihood < orientation_likelihood_threshold
            front_x_filtered = front_x.copy()
            back_x_filtered = back_x.copy()
            front_x_filtered[front_errors] = np.nan
            back_x_filtered[back_errors] = np.nan
            
            # Fill NaN values
            front_x_filled = front_x_filtered.ffill().bfill()
            back_x_filled = back_x_filtered.ffill().bfill()
            
            # If all values are NaN after filtering, use original data
            if front_x_filled.isna().all():
                front_x_filled = front_x
            if back_x_filled.isna().all():
                back_x_filled = back_x

            # Compute orientation vector (front - back) - points in direction animal is facing
            orientation_x = front_x_filled - back_x_filled
            
            # Smooth the orientation to reduce noise
            orientation_x_smooth = pd.Series(
                uniform_filter1d(orientation_x.values, size=orientation_smooth_window_size),
                index=orientation_x.index
            )
            
            # Get direction sign from orientation
            # Positive means front is to the right (forward in typical coordinate system)
            # Negative means front is to the left (backward)
            forward_sign = np.sign(orientation_x_smooth)
            forward_sign = forward_sign.ffill().bfill().fillna(1.0)
 
            # For each paw
            for rel_paw in rel_paws:
                # Extract relative x and y coordinates
                x = pose_data_with_rel[(scorer, individual, rel_paw, 'x')]
                y = pose_data_with_rel[(scorer, individual, rel_paw, 'y')]

                # Check if we have valid data
                if y.isna().all() or x.isna().all():
                    print(f'No valid data available for {rel_paw}')
                    continue

                # Smooth the signals to reduce noise
                x_smooth = pd.Series(
                    uniform_filter1d(x.values, size=smooth_window_size),
                    index=x.index
                )
                y_smooth = pd.Series(
                    uniform_filter1d(y.values, size=smooth_window_size),
                    index=y.index
                )

                # Transform x to body-relative coordinate system
                # Multiply by forward_sign to ensure "forward" is always positive
                # forward_sign changes over time to handle direction changes
                # This assumes the relative coordinates are already relative to body position
                x_rel_smooth = x_smooth * forward_sign

                # CRITERIA 1: The foot reaches its forward-most position
                x_range = np.nanmax(x_smooth) - np.nanmin(x_smooth)
                prominence_size = x_range * prominence_percentage

                forward_maxima, _ = find_peaks(
                    x_rel_smooth,
                    prominence=prominence_size,
                    distance=smooth_window_size
                )



                strikes[(scorer, individual)][rel_paw] = sorted(forward_maxima)

                if show_plots:
                    visualize_zeni_steps(
                        paw_name=rel_paw,
                        x=x,
                        y=y,
                        x_smooth=x_rel_smooth,
                        y_smooth=y_smooth,
                        forward_maxima=forward_maxima,
                        forward_sign=forward_sign
                    )

    return strikes

def visualize_zeni_steps(paw_name,
                         x,
                         y,
                         x_smooth,
                         y_smooth,
                         forward_maxima,
                         forward_sign,
                         figsize=(20,12)):
    # Create visualization
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f'{paw_name} - Simplified Zeni Algorithm Visualization', 
                fontsize=16, fontweight='bold', y=0.995)
    
    frames = x.index.values

    # Forward direction visualization
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(frames, forward_sign.values, 'g-', alpha=0.7, linewidth=1.5, label='Forward Direction')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.fill_between(frames, 0, forward_sign.values, where=(forward_sign.values > 0), 
                     color='green', alpha=0.2, label='Forward')
    ax1.fill_between(frames, 0, forward_sign.values, where=(forward_sign.values < 0), 
                     color='red', alpha=0.2, label='Backward')
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Direction Sign')
    ax1.set_title('Body Motion Direction')
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['Backward', 'Stationary', 'Forward'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # CRITERIA 1: The foot reaches its forward-most position
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(frames, x.values, 'gray', alpha=0.3, linewidth=1, label='Original X')
    ax2.plot(frames, x_smooth.values, 'b-', alpha=0.7, linewidth=1.5, label='Smoothed X')

    ax2.scatter(frames[forward_maxima], x_smooth.values[forward_maxima], color='red', s=50, 
                marker='v', alpha=0.5, zorder=3, label=f'Forward maxima ({len(forward_maxima)})')

    ax2.set_xlabel('Frame')
    ax2.set_ylabel('X Position (pixels)')
    ax2.set_title('Criterion 1: Forward-most Position')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    

    plt.show()
