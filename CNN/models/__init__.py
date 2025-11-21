"""
CNN Models Package

Contains custom and transfer learning architectures for vehicle detection.
"""

from .architectures import (
    MobileNetInspired,
    SqueezeNetInspired,
    ResNetInspired,
    TransferLearningModel,
    LSTMDistanceEstimator,
    create_model
)

__all__ = [
    'MobileNetInspired',
    'SqueezeNetInspired',
    'ResNetInspired',
    'TransferLearningModel',
    'LSTMDistanceEstimator',
    'create_model'
]
