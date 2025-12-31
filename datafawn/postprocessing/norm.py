"""
Normalization utilities for pose data.

This module provides functions for converting absolute coordinates to relative
coordinates, which are useful for event detection algorithms.
"""

import pandas as pd
from typing import Union


def get_coords(df: pd.DataFrame, bodypart: str, coord: str, scorer: str, individual: str) -> pd.Series:
    """Extract coordinates for a specific body part."""
    # Use .xs() for efficient MultiIndex indexing, then select the coordinate
    coords = df.xs((scorer, individual, bodypart), level=[0, 1, 2], axis=1)[coord]

    return coords

def paw_to_relative_position(df: pd.DataFrame, append_to_df: bool = True) -> pd.DataFrame:
    """
    Convert paw coordinates to relative position coordinates.
    
    Processes all combinations of scorers and individuals in the DataFrame.
    """
    # Create new dataframe with same structure as original, including relative positions as new bodyparts
    # Start with a copy of the original data
    df_with_rel = df.copy()

    # Get all unique (scorer, individual) combinations
    scorers = df.columns.get_level_values(0).unique()
    individuals = df.columns.get_level_values(1).unique()

    front_paws = ['front_left_paw', 'front_right_paw']
    back_paws = ['back_left_paw', 'back_right_paw']
    front_ref = 'back_base'
    back_ref = 'tail_base'

    # Process each (scorer, individual) combination
    for scorer in scorers:
        for individual in individuals:
            try:
                # Get reference points for this (scorer, individual) combination
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
            except (KeyError, IndexError):
                # Skip if this (scorer, individual) combination doesn't have required bodyparts
                continue

    # Sort columns lexicographically to maintain proper MultiIndex order
    df_with_rel = df_with_rel.sort_index(axis=1)

    print("Dataframe with relative positions created:")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_with_rel.shape}")
    print(f"\nNew bodyparts added:")
    # Use numeric index (level 2) for bodyparts, matching error_mask.py pattern
    new_bodyparts = [bp for bp in df_with_rel.columns.get_level_values(2).unique() 
                    if bp not in df.columns.get_level_values(2).unique()]
    print(new_bodyparts)
    
    if append_to_df:
        return df_with_rel
    else:
        # Extract only the relative position columns
        # Need to construct full MultiIndex tuples: (scorer, individual, bodypart, coord)
        cols = []
        for scorer in scorers:
            for individual in individuals:
                for bodypart in new_bodyparts:
                    for coord in ['x', 'y', 'likelihood']:
                        col_tuple = (scorer, individual, bodypart, coord)
                        if col_tuple in df_with_rel.columns:
                            cols.append(col_tuple)
        return df_with_rel[cols]