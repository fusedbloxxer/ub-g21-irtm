from jax import Array
from typing import TypedDict, Optional, List, Tuple, Literal, TypeAlias
from pandas import DataFrame
from dataclasses import dataclass


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
    'emotion': Optional[str],
    'video_name': Optional[str],
})
EmotionCauseConversation = TypedDict('EmotionCauseConversation', {
    'conversation_ID': int,
    'conversation': List[EmotionCauseUtterance],
    'emotion-cause_pairs': Optional[List[Tuple[str, str]]]
})


# Intermediary Dataset Representation
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
    cause_mask: Optional[Array]
    cause_span: Optional[Array]
    cause_labels: Optional[Array]
    emotion_labels: Optional[Array]


# Final Dataset Representation
@dataclass
class EmotionCauseData(object):
    metadata: EmotionCauseMetaData
    encoding: EmotionCauseEncoding
