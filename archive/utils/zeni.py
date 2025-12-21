import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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