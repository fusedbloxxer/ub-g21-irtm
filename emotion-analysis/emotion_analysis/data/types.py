from jax import Array
from typing import TypedDict, Optional, List, Tuple, Literal, TypeAlias, NotRequired
from pandas import DataFrame
from dataclasses import dataclass


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


# Types from JSON File
EmotionCauseUtterance = TypedDict('EmotionCauseUtterance', {
    'utterance_ID': int,
    'text': str,
    'speaker': str,
    'emotion': NotRequired[str],
    'video_name': NotRequired[str],
})
EmotionCauseConversation = TypedDict('EmotionCauseConversation', {
    'conversation_ID': int,
    'conversation': List[EmotionCauseUtterance],
    'emotion-cause_pairs': NotRequired[List[Tuple[str, str]]]
})
ECACData = List[EmotionCauseConversation]


# Metadata Representation
@dataclass
class EmotionCauseMetaData(object):
    # Data
    conversation: DataFrame

    # Labels
    emotion_cause_pairs: Optional[DataFrame] = None


# Dataset Encoding Representation
class EmotionCauseEncoding(TypedDict):
    # Data
    input_ids: Array
    input_mask: Array
    attention_mask: Array
    offset_mapping: Array

    # Labels
    cause_mask: NotRequired[Array]
    cause_span: NotRequired[Array]
    cause_labels: NotRequired[Array]
    emotion_labels: NotRequired[Array]


# Final Dataset Representation
@dataclass
class EmotionCauseData(object):
    metadata: EmotionCauseMetaData
    encoding: EmotionCauseEncoding
