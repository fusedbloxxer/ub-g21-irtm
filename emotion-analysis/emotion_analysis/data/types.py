from typing import TypedDict, Sequence, Optional, List, Dict, Tuple, Literal, TypeAlias, NotRequired
from dataclasses import dataclass
from pathlib import Path
from jax import Array
import numpy as np


# The competition consiers two tasks
SubTask: TypeAlias = Literal[
    '1',
    '2',
]


# There are three data splits in the competition
DataSplit: TypeAlias = Literal[
    'train',
    'trial',
    'test',
]


TrainSplit: TypeAlias = Literal[
    'train',
    'valid',
    'test',
]


# Paul Ekman Basic Emotion Categories
Emotion: TypeAlias = Literal[
    'neutral',
    'anger',
    'surprise',
    'sadness',
    'joy',
    'disgust',
    'fear'
]


DataStats = TypedDict('DataStats', {
    'emotion_labels': List[int],
})


# Common Dataset Types
EmotionCauseUtterance = TypedDict('EmotionCauseUtterance', {
    # From JSON
    'utterance_ID': int,
    'text': str,
    'speaker': str,
    'emotion': NotRequired[str],
    'video_name': NotRequired[str],

    # Filled-in values
    'video_path': NotRequired[Path],
    'emotion_label': NotRequired[int],
})
EmotionCauseConversation = TypedDict('EmotionCauseConversation', {
    # From JSON
    'conversation_ID': int,
    'conversation': List[EmotionCauseUtterance],
    'emotion-cause_pairs': NotRequired[List[Tuple[str, str]]],
    
    # Filled-in values
    'has_spans': bool,
    'has_video': bool,
    'has_causes': bool,
    'has_emotions': bool,
})
ECACData = List[EmotionCauseConversation]


# Dataset Encoding Representation
class EmotionCauseEncoding(TypedDict):
    # Data
    input_ids: Array
    conv_attn_mask: Array
    uttr_attn_mask: Array

    # Labels
    span_mask: NotRequired[Array]
    cause_span: NotRequired[Array]
    cause_labels: NotRequired[Array]
    emotion_labels: NotRequired[Array]

    # Weights for imbalanced data
    emotion_weight: NotRequired[Array]
