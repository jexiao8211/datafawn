import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional


def plot_bodyparts_position(pose_data: pd.DataFrame,
                            bodyparts: List[str],
                            scorer: Optional[str] = None,
                            individual: Optional[str] = None,
                            min_likelihood: float = 0.0,
                            figsize: tuple = (15, 5)):
    """
    Plot X and Y position over time for multiple bodyparts.
    
    Parameters:
    -----------
    pose_data : pd.DataFrame
        DataFrame with pose data (MultiIndex columns: scorer, individual, bodypart, coords)
    bodyparts : list of str
        List of bodypart names to plot (e.g., ['front_left_paw', 'back_base'])
    scorer : str, optional
        Scorer name. If None, uses the first scorer in the DataFrame.
    individual : str, optional
        Individual name. If None, uses the first individual in the DataFrame.
    min_likelihood : float, default=0.0
        Minimum likelihood threshold (filter out low-confidence detections)
    figsize : tuple, default=(15, 5)
        Figure size for the plots
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get scorer and individual if not provided
    if scorer is None:
        scorer = pose_data.columns.get_level_values(0).unique()[0]
    if individual is None:
        individual = pose_data.columns.get_level_values(1).unique()[0]
    
    # Create figure with subplots: one row per bodypart, two columns (X and Y)
    n_bodyparts = len(bodyparts)
    fig, axes = plt.subplots(n_bodyparts, 2, figsize=(figsize[0], figsize[1] * n_bodyparts))
    
    # Handle case where there's only one bodypart (axes becomes 1D)
    if n_bodyparts == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each bodypart
    for idx, bodypart in enumerate(bodyparts):
        try:
            # Extract coordinates using MultiIndex slicing
            x_data = pose_data[(scorer, individual, bodypart, 'x')]
            y_data = pose_data[(scorer, individual, bodypart, 'y')]
            likelihood_data = pose_data[(scorer, individual, bodypart, 'likelihood')]
            
            # Get frames
            frames = x_data.index.values
            
            # Filter by likelihood
            valid_mask = likelihood_data.values >= min_likelihood
            x_valid = x_data.values[valid_mask]
            y_valid = y_data.values[valid_mask]
            frames_valid = frames[valid_mask]
            
            # Plot X position
            axes[idx, 0].plot(frames_valid, x_valid, 'b-', linewidth=1.5, alpha=0.7, label='x position')
            axes[idx, 0].set_xlabel('Frame')
            axes[idx, 0].set_ylabel('X Position (pixels)')
            axes[idx, 0].set_title(f'{bodypart} - X Position Over Time\n({scorer}/{individual})')
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].legend()
            
            # Plot Y position
            axes[idx, 1].plot(frames_valid, y_valid, 'r-', linewidth=1.5, alpha=0.7, label='y position')
            axes[idx, 1].set_xlabel('Frame')
            axes[idx, 1].set_ylabel('Y Position (pixels)')
            axes[idx, 1].set_title(f'{bodypart} - Y Position Over Time\n({scorer}/{individual})')
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].legend()
            
        except (KeyError, IndexError) as e:
            # If bodypart not found, show error message in both subplots
            axes[idx, 0].text(0.5, 0.5, f'Bodypart "{bodypart}" not found', 
                             ha='center', va='center', transform=axes[idx, 0].transAxes,
                             fontsize=12, color='red')
            axes[idx, 0].set_title(f'{bodypart} - Not Found')
            axes[idx, 1].text(0.5, 0.5, f'Bodypart "{bodypart}" not found', 
                             ha='center', va='center', transform=axes[idx, 1].transAxes,
                             fontsize=12, color='red')
            axes[idx, 1].set_title(f'{bodypart} - Not Found')
    
    plt.tight_layout()
    return fig

