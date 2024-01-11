from typing import Any, TypedDict, List, Literal, Dict, cast
from torch.utils.data import Dataset
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pathlib as pb
import pandas as pd
from pandas import DataFrame
import json
import os
import re
import jax
from jax import Array
import jax.numpy as jnp

from .types import EmotionCauseConversation
from. types import EmotionCauseEncoding
from .types import EmotionCauseData
from .types import Emotion


@dataclass
class Utterance(object):
    uterrance_id: int
    text: str
    speaker: str
    emotion: Emotion | None = None
    video_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict, conversation_id: int, video_dir: pb.Path | None) -> 'Utterance':
        # Some metadata files have missing video annotations but we can compute them dynamically
        video_path = str(video_dir / f'dia{conversation_id}utt{data["utterance_ID"]}.mp4') if video_dir else None

        # Make sure the video exists on disk
        if video_path:
            assert os.path.exists(video_path), f'no video file was found on disk at {video_path}'

        # Mapping to internal format
        return Utterance(
            video_path=video_path,
            emotion=data.get('emotion', None),
            uterrance_id=data['utterance_ID'],
            speaker=data['speaker'],
            text=data['text'],
        )


@dataclass
class Conversation(object):
    conversation_id: int
    conversation: List[Utterance]
    emotion_cause: List[List[str]]

    @classmethod
    def from_dict(cls, data: dict, video_dir: pb.Path | None) -> 'Conversation':
        # All utterances in a conversation originate from itself
        make_utterance = partial(Utterance.from_dict, conversation_id=data['conversation_ID'], video_dir=video_dir)

        # Mapping to internal format
        return Conversation(
            conversation=list(map(make_utterance, data['conversation'])),
            emotion_cause=data.get('emotion-cause_pairs', []),
            conversation_id=data['conversation_ID'],
        )


class ECACDataset(Dataset):
    def __init__(self,
                 data_dir: pb.Path,
                 task: Literal['task_1', 'task_2'] = 'task_1',
                 split: Literal['train', 'eval', 'test'] = 'train') -> None:
        super().__init__()
        assert task in ('task_1', 'task_2'), 'invalid task type'
        assert split in ('train', 'eval', 'test'), 'invalid split type'

        # Data paths
        self.video_pattern: str = r'dia\d+utt\d+\.mp4'
        self.data_dir_path = data_dir / split / task
        self.data_path = self.data_dir_path / f'{task}_{split}.json'
        self.video_dir_path = self.data_dir_path / 'video'

        # Dataset settings
        self.task: Literal['task_1', 'task_2'] = task
        self.split: Literal['train', 'eval', 'test'] = split
        self.emotions: List[str] = ['neutral', 'anger', 'surprise', 'sadness', 'joy', 'disgust', 'fear']
        self.emotion_to_class = {e: i for i, e in enumerate(self.emotions)}
        self.class_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.num_emotions = len(self.emotions)

        # Integrity checks
        if not self.data_dir_path.exists():
            raise FileNotFoundError(self.data_dir_path)
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)
        if self.task == 'task_2':
            if not self.video_dir_path.exists():
                raise FileNotFoundError(self.video_dir_path)
            if len(video_paths := list(self.video_dir_path.iterdir())) == 0:
                raise Exception('no files were found under ' + str(self.video_dir_path))
            if not all(map(lambda x: re.fullmatch(self.video_pattern, x.name), video_paths)):
                raise Exception('found invalid files under ' + str(self.video_dir_path))

        # Read and prepare the dataset
        data = self.__read_data(self.data_path)
        data = self.__transform_data(data)
        self.data = data
        # self.data = self.__tokenize_data(self.data)

    # def __getitem__(self, index: int):
    #     data = self.data_[self.data_['conversation_id'] == index]
    #     if self.split == 'train':
    #         assert self.labels_ is not None, 'no labels were found for train split'
    #         labels = self.labels_[self.labels_['conversation_id'] == index]
    #     else:
    #         labels = None
    #     return (data, labels)

    # def __len__(self) -> int:
    #     return len(self.data_['conversation_id'].unique())

    def __read_data(self, path: Path) -> List[EmotionCauseConversation]:
        with open(path, 'r') as dataset_file:
            return json.load(dataset_file)

    def __transform_data(self, dataset: List[EmotionCauseConversation]) -> EmotionCauseData:
        # Transform to pandas format for faster processing
        data_, labels_ = [], []

        # Aggregate all conversations in two separate DFs
        for conversation in dataset:
            for utterance in conversation['conversation']:
                # First add data that is always present
                data_.append({
                    'conversation_id': conversation['conversation_ID'] - 1,
                    'utterance_id': utterance['utterance_ID'] - 1,
                    'speaker': utterance['speaker'],
                    'text': utterance['text'],
                })
                
                # Obtain utterance emotion
                if 'emotion' in utterance:
                    emotion: str = cast(str, utterance.get('emotion', ''))
                    emotion_label: int | None = self.emotion_to_class.get(emotion, None)
                    data_[-1]['emotion'] = emotion_label

                # Obtain video_path 
                if self.task == 'task_2':
                    video_name = f'dia{conversation["conversation_ID"]}utt{utterance["utterance_ID"]}.mp4'
                    video_path = self.video_dir_path / video_name
                    assert os.path.exists(video_path), f'no video file was found on disk at {video_path}'
                    data_[-1]['video_path'] = video_path

            # Proceed further only if labels are available
            if self.split != 'train':
                continue
            if not ('emotion-cause_pairs' in conversation and conversation['emotion-cause_pairs']):
                continue

            # Retireve all causal labels
            for emotion, cause in conversation['emotion-cause_pairs']:
                # First add data that is always present
                labels_.append({
                    'conversation_id': conversation['conversation_ID'] - 1,
                })

                # Obtain emotion
                utterance_id: int = int(emotion.split('_')[0]) - 1
                emotion_str: str = ''.join(emotion.split('_')[1:])
                label: int = self.emotion_to_class[emotion_str]
                labels_[-1]['utterance_id'] = utterance_id
                labels_[-1]['emotion'] = label

                # Obtain cause
                cause_id: int = int(cause.split('_')[0]) - 1
                labels_[-1]['cause_id'] = cause_id

                # Find the indices of the cause for task 1 only
                if self.task == 'task_1':
                    cause_str: str = ''.join(cause.split('_')[1:])
                    cause_index_start: int = conversation['conversation'][cause_id]['text'].find(cause_str)
                    cause_index_stop: int = cause_index_start + len(cause_str) - 1
                    assert cause_index_start != -1, 'could not find cause range (conv_id: {}, utt_id: {}, cause_id: {}, cause_str: {})' \
                        .format(conversation['conversation_ID'], utterance_id, cause_id, cause_str)
                    labels_[-1]['cause_start'] = cause_index_start
                    labels_[-1]['cause_stop'] = cause_index_stop

        # Aggregate data and labels
        data: DataFrame = pd.DataFrame(data_)
        labels: DataFrame | None = pd.DataFrame(labels_) if self.split == 'train' else None
        return EmotionCauseData(data, labels)

    def __tokenize_data(self, dataset: EmotionCauseData) -> EmotionCauseEncoding:
        pass
