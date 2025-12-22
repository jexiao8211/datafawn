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

from datafawn.pose_estimators import (
    DeepLabCutPoseEstimator
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

    'DeepLabCutPoseEstimator',

    'RelativePawPositionPostprocessor',
    'ErrorPostprocessor',

    'ZeniExtractor'
]

