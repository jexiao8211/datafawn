import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# def zeni_algorithm_vis(pose_data_with_rel, 
#                        window_size=5,
#                        min_contact_duration=3,
#                        velocity_threshold=10,
#                        prominence_percentage=0.05,
#                        figsize=(20, 12),
#                        show_plots=True):
#     """
#     Zeni algorithm with step-by-step visualizations for each paw.
    
#     This function performs the same algorithm as zeni_algorithm but creates
#     comprehensive visualizations showing the results of each processing step.
    
#     Processes all combinations of scorers and individuals in the DataFrame automatically.
    
#     Parameters:
#     -----------
#     pose_data_with_rel : pd.DataFrame
#         DataFrame with relative positions (from pose_data_with_rel)
#     window_size : int, default=5
#         Window size for finding local maxima (should be odd)
#     min_contact_duration : int, default=3
#         Minimum number of consecutive frames for a valid contact
#     velocity_threshold : float, default=10
#         Maximum velocity (pixels/frame) at contact point
#     error_mask : pd.DataFrame
#         DataFrame with boolean columns indicating problematic frames for each bodypart.
#         True values indicate frames with errors that should be filtered out.
#         Must have columns matching paw names: 'front_left_paw', 'front_right_paw',
#         'back_left_paw', 'back_right_paw'
#     prominence_percentage : float, default=0.05
#         Percentage of the signal range to use as the prominence threshold
#     figsize : tuple, default=(20, 12)
#         Figure size for the visualization plots
#     show_plots : bool, default=True
#         If True, displays plots immediately. If False, returns figure objects.
    
#     Returns:
#     --------
#     strikes : dict
#         Nested dictionary: strikes[(scorer, individual)][paw_name] contains
#         a list of frame indices where foot strikes occur for that combination.
#         Structure: {(scorer, individual): {paw_name: [frame_indices]}}
#     figs : dict
#         Nested dictionary: figs[(scorer, individual)][paw_name] contains
#         a matplotlib figure object for that combination.
#         Structure: {(scorer, individual): {paw_name: figure}}
#     """
#     from scipy.signal import find_peaks
#     from scipy.ndimage import uniform_filter1d
    
#     # Get all unique (scorer, individual) combinations
#     scorers = pose_data_with_rel.columns.get_level_values(0).unique()
#     individuals = pose_data_with_rel.columns.get_level_values(1).unique()
    
#     # Define paws and their relative bodypart names
#     paws = {
#         'front_left_paw': 'front_left_paw_rel',
#         'front_right_paw': 'front_right_paw_rel',
#         'back_left_paw': 'back_left_paw_rel',
#         'back_right_paw': 'back_right_paw_rel'
#     }
    
#     strikes = {}
#     figs = {}
    
#     # Process each (scorer, individual) combination
#     for scorer in scorers:
#         for individual in individuals:
#             # Initialize nested dictionaries for this combination
#             if (scorer, individual) not in strikes:
#                 strikes[(scorer, individual)] = {}
#             if (scorer, individual) not in figs:
#                 figs[(scorer, individual)] = {}
            
#             for paw_name, rel_bodypart in paws.items():
#                 try:
#                     # Extract relative y-coordinate (vertical position)
#                     y_rel = pose_data_with_rel[(scorer, individual, rel_bodypart, 'y')]
#                 except (KeyError, IndexError):
#                     # Skip if this (scorer, individual) combination doesn't have this bodypart
#                     strikes[(scorer, individual)][paw_name] = []
#                     continue
                                
#                 # Smooth the signal to reduce noise (convert to numpy array first)
#                 y_smooth = pd.Series(
#                     uniform_filter1d(y_rel.values, size=window_size),
#                     index=y_rel.index
#                 )
                
#                 # Calculate velocity (change in y per frame)
#                 velocity = np.abs(y_smooth.diff())
                
#                 # Find local maxima in y (lowest vertical position = foot strike)
#                 # Use prominence to avoid detecting small fluctuations
#                 # Prominence = minimum height difference from surrounding peaks
#                 y_array = y_smooth.values
#                 valid_indices = ~np.isnan(y_array)
        
#                 if np.sum(valid_indices) < window_size:
#                     # Not enough valid data
#                     strikes[(scorer, individual)][paw_name] = []
#                     # Create empty figure with message
#                     fig = plt.figure(figsize=figsize)
#                     fig.suptitle(f'{scorer}/{individual} - {paw_name} - Insufficient Data', fontsize=16, fontweight='bold')
#                     plt.text(0.5, 0.5, f'Not enough valid data (need at least {window_size} frames)', 
#                             ha='center', va='center', fontsize=14, transform=fig.transFigure)
#                     figs[(scorer, individual)][paw_name] = fig
#                     if show_plots:
#                         plt.show()
#                     continue
        
#                 # Calculate prominence threshold (e.g., 5% of signal range)
#                 y_range = np.nanmax(y_array) - np.nanmin(y_array)
#                 prominence_size = y_range * prominence_percentage  
                
#                 # Find peaks (local maxima)
#                 peaks, properties = find_peaks(
#                     y_array,
#                     prominence=prominence_size,
#                     distance=window_size  # Minimum distance between peaks
#                 )
                
#                 # Track peaks at different filtering stages
#                 peaks_after_velocity = []
#                 peaks_after_error = []
#                 peaks_after_duration = []
                
                
#                 # Filter peaks by velocity and error_mask
#                 valid_strikes = []
#                 for peak_idx in peaks:
#                     frame_idx = y_smooth.index[peak_idx]
                    
#                     # Check velocity at contact point (should be low)
#                     vel_passed = True
#                     if peak_idx < len(velocity):
#                         vel_at_contact = velocity.iloc[peak_idx]
#                         # Skip if velocity is NaN (first frame) or exceeds threshold
#                         if pd.isna(vel_at_contact) or vel_at_contact > velocity_threshold:
#                             vel_passed = False
#                     else:
#                         vel_passed = False
                    
#                     if vel_passed:
#                         peaks_after_velocity.append(peak_idx)
                    
#                     if not vel_passed:
#                         continue  # Too much movement, not a stable contact
                    
                    
                    
                    
#                     # Check if this is part of a contact duration
#                     # Look for consecutive frames around the peak with similar y values
#                     contact_window = window_size
#                     start_idx = max(0, peak_idx - contact_window // 2)
#                     end_idx = min(len(y_smooth), peak_idx + contact_window // 2 + 1)
                    
#                     window_y = y_smooth.iloc[start_idx:end_idx]
                    
#                     # Count valid frames in window (no errors and not NaN)
#                     valid_in_window = np.sum(
#                         (~np.isnan(window_y))
#                     )
                    
#                     if valid_in_window >= min_contact_duration:
#                         peaks_after_duration.append(peak_idx)
#                         valid_strikes.append(int(frame_idx))
                
#                 strikes[(scorer, individual)][paw_name] = sorted(valid_strikes)
        
#                 # Create comprehensive visualization
#                 fig = plt.figure(figsize=figsize)
#                 gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
#                 fig.suptitle(f'{scorer}/{individual} - {paw_name} - Zeni Algorithm Step-by-Step Visualization', 
#                             fontsize=16, fontweight='bold', y=0.995)
        
#                 # Get frame indices for plotting
#                 frames = y_rel.index.values
                
#                 # Plot 1: Original y position
#                 ax1 = fig.add_subplot(gs[0, 0])
#                 ax1.plot(frames, y_rel.values, 'b-', alpha=0.7, linewidth=1.5, label='Original Y position')
#                 ax1.set_xlabel('Frame')
#                 ax1.set_ylabel('Y Position (pixels)')
#                 ax1.set_title('Step 1: Original Y Position')
#                 ax1.grid(True, alpha=0.3)
#                 ax1.legend()
                
#                 # Plot 4: After smoothing
#                 ax4 = fig.add_subplot(gs[1, 0])
#                 ax4.plot(frames, y_rel.values, 'gray', alpha=0.3, linewidth=1, label='Before smoothing')
#                 ax4.plot(frames, y_smooth.values, 'purple', alpha=0.7, linewidth=2, label='After smoothing')
#                 ax4.set_xlabel('Frame')
#                 ax4.set_ylabel('Y Position (pixels)')
#                 ax4.set_title(f'Step 4: After Smoothing (window={window_size})')
#                 ax4.grid(True, alpha=0.3)
#                 ax4.legend()
                
#                 # Plot 5: Velocity
#                 ax5 = fig.add_subplot(gs[1, 1])
#                 ax5.plot(frames[1:], velocity.iloc[1:].values, 'orange', alpha=0.7, 
#                         linewidth=1.5, label='Velocity')
#                 ax5.axhline(y=velocity_threshold, color='red', linestyle='--', 
#                            linewidth=2, label=f'Threshold ({velocity_threshold})')
#                 ax5.set_xlabel('Frame')
#                 ax5.set_ylabel('Velocity (pixels/frame)')
#                 ax5.set_title('Step 5: Velocity Calculation')
#                 ax5.grid(True, alpha=0.3)
#                 ax5.legend()
                
#                 # Plot 6: Detected peaks (before filtering)
#                 ax6 = fig.add_subplot(gs[1, 2])
#                 ax6.plot(frames, y_smooth.values, 'purple', alpha=0.7, linewidth=1.5, label='Smoothed Y')
#                 if len(peaks) > 0:
#                     peak_frames = frames[peaks]
#                     peak_y = y_smooth.iloc[peaks]
#                     ax6.scatter(peak_frames, peak_y.values, color='red', s=100, 
#                                marker='v', zorder=5, label=f'All peaks ({len(peaks)})')
#                 ax6.set_xlabel('Frame')
#                 ax6.set_ylabel('Y Position (pixels)')
#                 ax6.set_title(f'Step 6: Detected Peaks\n(prominence={prominence_size:.2f})')
#                 ax6.grid(True, alpha=0.3)
#                 ax6.legend()
                
#                 # Plot 7: Peaks after velocity filtering
#                 ax7 = fig.add_subplot(gs[2, 0])
#                 ax7.plot(frames, y_smooth.values, 'purple', alpha=0.5, linewidth=1, label='Smoothed Y')
#                 if len(peaks) > 0:
#                     peak_frames = frames[peaks]
#                     peak_y = y_smooth.iloc[peaks]
#                     ax7.scatter(peak_frames, peak_y.values, color='gray', s=50, 
#                                marker='v', alpha=0.5, zorder=3, label='All peaks')
#                 if len(peaks_after_velocity) > 0:
#                     vel_peak_frames = frames[peaks_after_velocity]
#                     vel_peak_y = y_smooth.iloc[peaks_after_velocity]
#                     ax7.scatter(vel_peak_frames, vel_peak_y.values, color='orange', s=100, 
#                                marker='v', zorder=5, label=f'After velocity filter ({len(peaks_after_velocity)})')
#                 ax7.set_xlabel('Frame')
#                 ax7.set_ylabel('Y Position (pixels)')
#                 ax7.set_title('Step 7: After Velocity Filtering')
#                 ax7.grid(True, alpha=0.3)
#                 ax7.legend(fontsize=8)
                
                
#                 # Plot 9: Final valid strikes
#                 ax9 = fig.add_subplot(gs[2, 2])
#                 ax9.plot(frames, y_smooth.values, 'purple', alpha=0.5, linewidth=1, label='Smoothed Y')
#                 if len(peaks_after_duration) > 0:
#                     dur_peak_frames = frames[peaks_after_duration]
#                     dur_peak_y = y_smooth.iloc[peaks_after_duration]
#                     ax9.scatter(dur_peak_frames, dur_peak_y.values, color='green', s=150, 
#                                marker='v', zorder=5, edgecolors='black', linewidths=1.5,
#                                label=f'Final strikes ({len(valid_strikes)})')
#                 ax9.set_xlabel('Frame')
#                 ax9.set_ylabel('Y Position (pixels)')
#                 ax9.set_title(f'Step 9: Final Valid Strikes\n(min_duration={min_contact_duration})')
#                 ax9.grid(True, alpha=0.3)
#                 ax9.legend()
                
#                 # Add summary text
#                 summary_text = (
#                     f"Summary:\n"
#                     f"  Total peaks detected: {len(peaks)}\n"
#                     f"  After velocity filter: {len(peaks_after_velocity)}\n"
#                     f"  After error filter: {len(peaks_after_error)}\n"
#                     f"  After duration filter: {len(peaks_after_duration)}\n"
#                     f"  Final valid strikes: {len(valid_strikes)}\n"
#                     f"  Prominence threshold: {prominence_size:.2f}\n"
#                     f"  Velocity threshold: {velocity_threshold}"
#                 )
#                 fig.text(0.02, 0.02, summary_text, fontsize=10, 
#                         verticalalignment='bottom', family='monospace',
#                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
#                 figs[(scorer, individual)][paw_name] = fig
                
#                 if show_plots:
#                     plt.show()
    
#     return strikes

    
def zeni_algorithm_vis(pose_data_with_rel, 
                       window_size=5,
                       scorer=None,
                       individual=None,
                       min_contact_duration=3,
                       velocity_threshold=10,
                       error_mask=None,
                       prominence_percentage=0.05,
                       figsize=(20, 12),
                       show_plots=True):
    """
    Simplified Zeni algorithm for detecting foot strikes.
    
    Foot strike is detected when all three conditions are met:
    1. The foot reaches its lowest vertical position relative to the pelvis (local maximum in y, since y increases downward)
    2. The foot is moving downward (positive vertical velocity)
    3. Forward velocity of the foot decreases sharply (negative acceleration in x)
    
    Parameters:
    -----------
    pose_data_with_rel : pd.DataFrame
        DataFrame with relative positions (from pose_data_with_rel)
    window_size : int, default=5
        Window size for smoothing (should be odd)
    scorer : str, optional
        Scorer name (extracted from df if not provided)
    individual : str, optional
        Individual name (extracted from df if not provided)
    min_contact_duration : int, default=3
        Minimum number of consecutive frames for a valid contact (not used in simplified algorithm)
    velocity_threshold : float, default=10
        Threshold for detecting sharp decrease in forward velocity (not used in simplified algorithm)
    error_mask : pd.DataFrame, optional
        DataFrame with boolean columns indicating problematic frames for each bodypart.
        True values indicate frames with errors that should be filtered out.
        Must have columns matching paw names: 'front_left_paw', 'front_right_paw',
        'back_left_paw', 'back_right_paw'
    prominence_percentage : float, default=0.05
        Percentage of the signal range to use as the prominence threshold for finding local maxima
    figsize : tuple, default=(20, 12)
        Figure size for the visualization plots
    show_plots : bool, default=True
        If True, displays plots immediately. If False, returns figure objects without displaying
    
    Returns:
    --------
    strikes : dict
        Dictionary with paw names as keys and lists of frame indices as values
        where foot strikes occur
    figs : dict
        Dictionary with paw names as keys and matplotlib figure objects as values
    """
    # Define scorer and individual if not provided
    if scorer is None:
        scorer = pose_data_with_rel.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = pose_data_with_rel.columns.get_level_values(1).unique()[0]
    
    # Define paws and their relative bodypart names
    paws = {
        'front_left_paw': 'front_left_paw_rel',
        'front_right_paw': 'front_right_paw_rel',
        'back_left_paw': 'back_left_paw_rel',
        'back_right_paw': 'back_right_paw_rel'
    }
    ground_bodypart = 'back_base'
    
    strikes = {}
    figs = {}
    
    for paw_name, rel_bodypart in paws.items():
        # Extract relative x and y coordinates
        x_rel = pose_data_with_rel[(scorer, individual, rel_bodypart, 'x')]
        y_rel = pose_data_with_rel[(scorer, individual, rel_bodypart, 'y')]

        ground_x = pose_data_with_rel[(scorer, individual, ground_bodypart, 'x')]
        
        # Check if we have valid data
        if y_rel.isna().all() or x_rel.isna().all():
            print(f'No valid data available for {paw_name}')
            continue
        
        # Smooth the signals to reduce noise
        x_smooth = pd.Series(
            uniform_filter1d(x_rel.values, size=window_size),
            index=x_rel.index
        )
        y_smooth = pd.Series(
            uniform_filter1d(y_rel.values, size=window_size),
            index=y_rel.index
        )
        ground_x_smooth = pd.Series(
            uniform_filter1d(ground_x.values, size=window_size),
            index=ground_x.index
        )
        
        
        # Calculate velocities
        # Vertical velocity: positive means moving downward (y increases downward in image coordinates)
        y_velocity = y_smooth.diff()
        
        # Forward velocity: change in x position
        x_velocity = x_smooth.diff()
        ground_x_velocuty = ground_x_smooth.diff()
        forward_velocity = x_velocity * np.sign(ground_x_velocuty) # positive means moving forward relative to the animal's body

        # Forward acceleration: change in forward velocity (negative = deceleration)
        forward_acceleration = forward_velocity.diff()
         
        # Find local maxima in y (lowest vertical position = highest y value in image coordinates)
        y_array = y_smooth.values
        valid_indices = ~np.isnan(y_array)
        
        if np.sum(valid_indices) < window_size:
            print(f'No valid data available for {paw_name}')
            continue
        
        # Calculate prominence threshold
        y_range = np.nanmax(y_array) - np.nanmin(y_array)
        prominence_size = y_range * prominence_percentage
        
        # Find local maxima (lowest position = highest y)
        maxima, _ = find_peaks(
            y_array,
            prominence=prominence_size,
            distance=window_size
        )
        

        # Filter maxima based on the three criteria
        # Track which maxima pass each criterion for visualization
        maxima_all = maxima
        maxima_after_downward = []
        maxima_after_deceleration = []
        valid_strikes = []
        
        for max_idx in maxima:
            frame_idx = y_smooth.index[max_idx]
            
            # Skip if we're at the boundaries
            if max_idx < 2 or max_idx >= len(y_smooth) - 1:
                continue
            
            # Criterion 1: Foot is moving downward (positive vertical velocity)
            # Check velocity just before the maximum (approaching the ground)
            vel_before = y_velocity.iloc[max_idx - 1]
            if pd.isna(vel_before) or vel_before <= 0:
                continue  # Not moving downward
            maxima_after_downward.append(max_idx)
            
            # Criterion 2: Forward velocity decreases sharply (negative acceleration)
            # Check acceleration at the maximum
            accel_at_max = forward_acceleration.iloc[max_idx]
            if pd.isna(accel_at_max) or accel_at_max >= 0:
                continue  # Not decelerating sharply
            maxima_after_deceleration.append(max_idx)
            
            # Criterion 3: Foot reaches lowest vertical position (already satisfied by finding maxima)
            # Additional check: ensure this is actually a local maximum
            if max_idx > 0 and max_idx < len(y_smooth) - 1:
                if y_smooth.iloc[max_idx] < y_smooth.iloc[max_idx - 1] or \
                   y_smooth.iloc[max_idx] < y_smooth.iloc[max_idx + 1]:
                    continue  # Not actually a maximum
            
            # All criteria met - this is a valid strike
            valid_strikes.append(int(frame_idx))
        
        strikes[paw_name] = sorted(valid_strikes)

        figs[paw_name] = visualize_zeni_steps(
            paw_name=paw_name,
            y_rel=y_rel,
            y_smooth=y_smooth,
            y_velocity=y_velocity,
            forward_velocity=forward_velocity,
            forward_acceleration=forward_acceleration,
            maxima_all=maxima_all,
            maxima_after_downward=maxima_after_downward,
            maxima_after_deceleration=maxima_after_deceleration,
            valid_strikes=valid_strikes,
            prominence_size=prominence_size
        )

    
    return strikes


def visualize_zeni_steps(paw_name,
                        y_rel=None,
                        y_smooth=None,
                        y_velocity=None,
                        forward_velocity=None,
                        forward_acceleration=None,
                        maxima_all=None,
                        maxima_after_downward=None,
                        maxima_after_deceleration=None,
                        valid_strikes=None,
                        prominence_size=None,
                        figsize=(20, 12),
                        show_plots=True):

    # Create visualization
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f'{paw_name} - Simplified Zeni Algorithm Visualization', 
                fontsize=16, fontweight='bold', y=0.995)
    
    frames = y_rel.index.values
    
    # Plot 1: Vertical position with detected strikes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(frames, y_rel.values, 'gray', alpha=0.3, linewidth=1, label='Original Y')
    ax1.plot(frames, y_smooth.values, 'b-', alpha=0.7, linewidth=1.5, label='Smoothed Y')
    
    # Mark all detected maxima
    if len(maxima_all) > 0:
        max_frames = frames[maxima_all]
        max_y = y_smooth.iloc[maxima_all]
        ax1.scatter(max_frames, max_y.values, color='red', s=50, 
                    marker='v', alpha=0.5, zorder=3, label=f'All maxima ({len(maxima_all)})')
    
    # Mark final valid strikes
    if len(valid_strikes) > 0:
        strike_frames = np.array(valid_strikes)
        strike_y = y_smooth.loc[strike_frames]
        ax1.scatter(strike_frames, strike_y.values, color='green', s=150, 
                    marker='v', zorder=5, edgecolors='black', linewidths=1.5,
                    label=f'Valid strikes ({len(valid_strikes)})')
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Criterion 3: Lowest Vertical Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Plot 2: Vertical velocity (downward movement)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(frames[1:], vertical_velocity.iloc[1:].values, 'purple', 
            alpha=0.7, linewidth=1.5, label='Vertical velocity')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Mark frames where velocity is positive (moving downward)
    # Note: we check velocity at max_idx - 1, so plot that frame
    if len(maxima_after_downward) > 0:
        down_indices = [idx - 1 for idx in maxima_after_downward if idx > 0]
        down_frames = frames[down_indices]
        down_vel = vertical_velocity.iloc[down_indices]
        ax2.scatter(down_frames, down_vel.values, color='orange', s=100, 
                    marker='o', zorder=5, label=f'Downward movement ({len(maxima_after_downward)})')
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Vertical Velocity (pixels/frame)')
    ax2.set_title('Criterion 1: Foot Moving Downward')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Plot 3: Forward velocity
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(frames[1:], forward_velocity.iloc[1:].values, 'blue', 
            alpha=0.7, linewidth=1.5, label='Forward velocity')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Forward Velocity (pixels/frame)')
    ax3.set_title('Forward Velocity')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Forward acceleration (deceleration)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(frames[2:], forward_acceleration.iloc[2:].values, 'red', 
            alpha=0.7, linewidth=1.5, label='Forward acceleration')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Mark frames with negative acceleration (deceleration)
    if len(maxima_after_deceleration) > 0:
        decel_frames = frames[maxima_after_deceleration]
        decel_accel = forward_acceleration.iloc[maxima_after_deceleration]
        ax4.scatter(decel_frames, decel_accel.values, color='orange', s=100, 
                    marker='o', zorder=5, label=f'Sharp deceleration ({len(maxima_after_deceleration)})')
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Forward Acceleration (pixels/frame²)')
    ax4.set_title('Criterion 2: Forward Velocity Decreases Sharply')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # Add summary text
    summary_text = (
        f"Summary:\n"
        f"  Total maxima detected: {len(maxima_all)}\n"
        f"  After downward filter: {len(maxima_after_downward)}\n"
        f"  After deceleration filter: {len(maxima_after_deceleration)}\n"
        f"  Final valid strikes: {len(valid_strikes)}\n"
        f"  Prominence threshold: {prominence_size:.2f}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    
    if show_plots:
        plt.show()

    return fig


    
def zeni_algorithm(pose_data_with_rel, 
                   window_size=5,
                   min_contact_duration=3,
                   velocity_threshold=10,
                   error_mask=None,
                   prominence_percentage=0.05):
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
    prominence_percentage : float, default=0.05
        Percentage of the signal range to use as the prominence threshold
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
        prominence_size = y_range * prominence_percentage  
        
        # Find peaks (local maxima)
        peaks, properties = find_peaks(
            y_array,
            prominence=prominence_size,
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


def zeni_algorithm_vis(pose_data_with_rel, 
                       window_size=5,
                       min_contact_duration=3,
                       velocity_threshold=10,
                       error_mask=None,
                       prominence_percentage=0.05,
                       figsize=(20, 12),
                       show_plots=True):
    """
    Zeni algorithm with step-by-step visualizations for each paw.
    
    This function performs the same algorithm as zeni_algorithm but creates
    comprehensive visualizations showing the results of each processing step.
    
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
    prominence_percentage : float, default=0.05
        Percentage of the signal range to use as the prominence threshold
    figsize : tuple, default=(20, 12)
        Figure size for the visualization plots
    show_plots : bool, default=True
        If True, displays plots immediately. If False, returns figure objects.
    
    Returns:
    --------
    strikes : dict
        Dictionary with paw names as keys and lists of frame indices as values
        where foot strikes occur
    figs : dict
        Dictionary with paw names as keys and matplotlib figure objects as values
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
    figs = {}
    
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
            # Create empty figure with message
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f'{paw_name} - No Valid Data', fontsize=16, fontweight='bold')
            plt.text(0.5, 0.5, 'No valid data available for this paw', 
                    ha='center', va='center', fontsize=14, transform=fig.transFigure)
            figs[paw_name] = fig
            if show_plots:
                plt.show()
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
            # Create empty figure with message
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f'{paw_name} - Insufficient Data', fontsize=16, fontweight='bold')
            plt.text(0.5, 0.5, f'Not enough valid data (need at least {window_size} frames)', 
                    ha='center', va='center', fontsize=14, transform=fig.transFigure)
            figs[paw_name] = fig
            if show_plots:
                plt.show()
            continue
        
        # Calculate prominence threshold (e.g., 5% of signal range)
        y_range = np.nanmax(y_array) - np.nanmin(y_array)
        prominence_size = y_range * prominence_percentage  
        
        # Find peaks (local maxima)
        peaks, properties = find_peaks(
            y_array,
            prominence=prominence_size,
            distance=window_size  # Minimum distance between peaks
        )
        
        # Track peaks at different filtering stages
        peaks_after_velocity = []
        peaks_after_error = []
        peaks_after_duration = []
        
        # Filter peaks by velocity and error_mask
        valid_strikes = []
        for peak_idx in peaks:
            frame_idx = y_smooth.index[peak_idx]
            
            # Check velocity at contact point (should be low)
            vel_passed = True
            if peak_idx < len(velocity):
                vel_at_contact = velocity.iloc[peak_idx]
                # Skip if velocity is NaN (first frame) or exceeds threshold
                if pd.isna(vel_at_contact) or vel_at_contact > velocity_threshold:
                    vel_passed = False
            else:
                vel_passed = False
            
            if vel_passed:
                peaks_after_velocity.append(peak_idx)
            
            if not vel_passed:
                continue  # Too much movement, not a stable contact
            
            # Check error_mask at peak
            error_passed = not error_mask[paw_name].iloc[peak_idx]
            if error_passed:
                peaks_after_error.append(peak_idx)
            
            if not error_passed:
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
                peaks_after_duration.append(peak_idx)
                valid_strikes.append(int(frame_idx))
        
        strikes[paw_name] = sorted(valid_strikes)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle(f'{paw_name} - Zeni Algorithm Step-by-Step Visualization', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Get frame indices for plotting
        frames = y_rel.index.values
        
        # Plot 1: Original y position
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(frames, y_rel.values, 'b-', alpha=0.7, linewidth=1.5, label='Original Y position')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title('Step 1: Original Y Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: After error mask filtering
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(frames, y_rel.values, 'gray', alpha=0.3, linewidth=1, label='Original')
        ax2.plot(frames[valid_mask], y_valid[valid_mask].values, 'b-', alpha=0.7, 
                linewidth=1.5, label='Valid (after error mask)')
        # Mark invalid points
        invalid_frames = frames[~valid_mask]
        if len(invalid_frames) > 0:
            invalid_y = y_rel.loc[invalid_frames]
            ax2.scatter(invalid_frames, invalid_y.values, color='red', s=20, 
                       alpha=0.6, marker='x', label='Invalid (error mask)')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title('Step 2: After Error Mask Filtering')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Plot 3: After filling NaN values
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(frames, y_valid.values, 'gray', alpha=0.3, linewidth=1, label='Before fill')
        ax3.plot(frames, y_filled.values, 'g-', alpha=0.7, linewidth=1.5, label='After NaN fill')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Y Position (pixels)')
        ax3.set_title('Step 3: After Filling NaN Values')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: After smoothing
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(frames, y_filled.values, 'gray', alpha=0.3, linewidth=1, label='Before smoothing')
        ax4.plot(frames, y_smooth.values, 'purple', alpha=0.7, linewidth=2, label='After smoothing')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Y Position (pixels)')
        ax4.set_title(f'Step 4: After Smoothing (window={window_size})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Velocity
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(frames[1:], velocity.iloc[1:].values, 'orange', alpha=0.7, 
                linewidth=1.5, label='Velocity')
        ax5.axhline(y=velocity_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({velocity_threshold})')
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Velocity (pixels/frame)')
        ax5.set_title('Step 5: Velocity Calculation')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Detected peaks (before filtering)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(frames, y_smooth.values, 'purple', alpha=0.7, linewidth=1.5, label='Smoothed Y')
        if len(peaks) > 0:
            peak_frames = frames[peaks]
            peak_y = y_smooth.iloc[peaks]
            ax6.scatter(peak_frames, peak_y.values, color='red', s=100, 
                       marker='v', zorder=5, label=f'All peaks ({len(peaks)})')
        ax6.set_xlabel('Frame')
        ax6.set_ylabel('Y Position (pixels)')
        ax6.set_title(f'Step 6: Detected Peaks\n(prominence={prominence_size:.2f})')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7: Peaks after velocity filtering
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(frames, y_smooth.values, 'purple', alpha=0.5, linewidth=1, label='Smoothed Y')
        if len(peaks) > 0:
            peak_frames = frames[peaks]
            peak_y = y_smooth.iloc[peaks]
            ax7.scatter(peak_frames, peak_y.values, color='gray', s=50, 
                       marker='v', alpha=0.5, zorder=3, label='All peaks')
        if len(peaks_after_velocity) > 0:
            vel_peak_frames = frames[peaks_after_velocity]
            vel_peak_y = y_smooth.iloc[peaks_after_velocity]
            ax7.scatter(vel_peak_frames, vel_peak_y.values, color='orange', s=100, 
                       marker='v', zorder=5, label=f'After velocity filter ({len(peaks_after_velocity)})')
        ax7.set_xlabel('Frame')
        ax7.set_ylabel('Y Position (pixels)')
        ax7.set_title('Step 7: After Velocity Filtering')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)
        
        # Plot 8: Peaks after error mask filtering
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(frames, y_smooth.values, 'purple', alpha=0.5, linewidth=1, label='Smoothed Y')
        if len(peaks_after_velocity) > 0:
            vel_peak_frames = frames[peaks_after_velocity]
            vel_peak_y = y_smooth.iloc[peaks_after_velocity]
            ax8.scatter(vel_peak_frames, vel_peak_y.values, color='gray', s=50, 
                       marker='v', alpha=0.5, zorder=3, label='After velocity')
        if len(peaks_after_error) > 0:
            err_peak_frames = frames[peaks_after_error]
            err_peak_y = y_smooth.iloc[peaks_after_error]
            ax8.scatter(err_peak_frames, err_peak_y.values, color='blue', s=100, 
                       marker='v', zorder=5, label=f'After error filter ({len(peaks_after_error)})')
        ax8.set_xlabel('Frame')
        ax8.set_ylabel('Y Position (pixels)')
        ax8.set_title('Step 8: After Error Mask Filtering')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        # Plot 9: Final valid strikes
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(frames, y_smooth.values, 'purple', alpha=0.5, linewidth=1, label='Smoothed Y')
        if len(peaks_after_duration) > 0:
            dur_peak_frames = frames[peaks_after_duration]
            dur_peak_y = y_smooth.iloc[peaks_after_duration]
            ax9.scatter(dur_peak_frames, dur_peak_y.values, color='green', s=150, 
                       marker='v', zorder=5, edgecolors='black', linewidths=1.5,
                       label=f'Final strikes ({len(valid_strikes)})')
        ax9.set_xlabel('Frame')
        ax9.set_ylabel('Y Position (pixels)')
        ax9.set_title(f'Step 9: Final Valid Strikes\n(min_duration={min_contact_duration})')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        
        # Add summary text
        summary_text = (
            f"Summary:\n"
            f"  Total peaks detected: {len(peaks)}\n"
            f"  After velocity filter: {len(peaks_after_velocity)}\n"
            f"  After error filter: {len(peaks_after_error)}\n"
            f"  After duration filter: {len(peaks_after_duration)}\n"
            f"  Final valid strikes: {len(valid_strikes)}\n"
            f"  Prominence threshold: {prominence_size:.2f}\n"
            f"  Velocity threshold: {velocity_threshold}"
        )
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        figs[paw_name] = fig
        
        if show_plots:
            plt.show()
    
    return strikes, figs