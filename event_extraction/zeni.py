import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def zeni_algorithm(pose_data_with_rel,
                   smooth_window_size=5,
                   prominence_percentage=0.05,
                   show_plots=True):
    """
    Simplified Zeni algorithm for detecting foot strikes.
        
    Parameters:
    -----------
    pose_data_with_rel : pd.DataFrame
        DataFrame with relative positions (from pose_data_with_rel)
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

    ground_bodypart = 'back_base'


    strikes = {}
    # Process each (scorer, individual) combination
    for scorer in scorers:
        for individual in individuals:

            # Initialize nested dictionaries for this combination
            if (scorer, individual) not in strikes:
                strikes[(scorer, individual)] = {}

            # Get grounding data
            ground_x = pose_data_with_rel[(scorer, individual, ground_bodypart, 'x')]
            ground_x_smooth = pd.Series(
                uniform_filter1d(ground_x.values, size=smooth_window_size),
                index=ground_x.index
            )

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

                # CRITERIA 1: The foot reaches its forward-most position
                y_range = np.nanmax(x_smooth) - np.nanmin(x_smooth)
                prominence_size = y_range * prominence_percentage

                forward_maxima, _ = find_peaks(
                    x_smooth,
                    prominence=prominence_size,
                    distance=smooth_window_size
                )




                strikes[(scorer, individual)][rel_paw] = sorted(forward_maxima)

                if show_plots:
                    visualize_zeni_steps(
                        paw_name=rel_paw,
                        x=x,
                        y=y,
                        x_smooth=x_smooth,
                        y_smooth=y_smooth,
                        forward_maxima=forward_maxima
                    )

    return strikes

def visualize_zeni_steps(paw_name,
                         x,
                         y,
                         x_smooth,
                         y_smooth,
                         forward_maxima,
                         figsize=(20,12)):
    # Create visualization
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f'{paw_name} - Simplified Zeni Algorithm Visualization', 
                fontsize=16, fontweight='bold', y=0.995)
    
    frames = x.index.values

    # CRITERIA 1: The foot reaches its forward-most position
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(frames, x.values, 'gray', alpha=0.3, linewidth=1, label='Original X')
    ax1.plot(frames, x_smooth.values, 'b-', alpha=0.7, linewidth=1.5, label='Smoothed X')

    ax1.scatter(frames[forward_maxima], x_smooth.values[forward_maxima], color='red', s=50, 
                marker='v', alpha=0.5, zorder=3, label=f'Forward maxima ({len(forward_maxima)})')

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position (pixels)')
    ax1.set_title('Criterion 1: Forward-most Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    plt.show()
