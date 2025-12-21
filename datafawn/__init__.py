"""
Event Detection Pipeline

A modular pipeline system for pose estimation, postprocessing, and event extraction.
"""

from datafawn.pipeline import (
    EventDetectionPipeline,
    PoseEstimator,
    Postprocessor,
    EventExtractor
)

from datafawn.postprocessors import (
    RelativePawPositionPostprocessor,
    ErrorPostprocessor
)

from datafawn.event_extractors import (
    ZeniExtractor
)

__all__ = [
    'EventDetectionPipeline',
    'PoseEstimator',
    'Postprocessor',
    'EventExtractor',

    'RelativePawPositionPostprocessor',
    'ErrorPostprocessor',

    'ZeniExtractor'
]

