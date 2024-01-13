from typing import Any, TypedDict, Optional, List, Literal, Dict, Tuple, cast
from torch.utils.data import Dataset
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pathlib as pb
import pandas as pd
from pandas import DataFrame
from itertools import islice
import json
import os
import re
import jax
from jax import Array
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
import tqdm
import math

from .types import EmotionCauseConversation
from. types import EmotionCauseEncoding
from .types import EmotionCauseMetaData
from .types import EmotionCauseData
from .types import Emotion

from .transform import DataTransform
from .transform import DataTokenize


class ECACDataset(Dataset):
    def __init__(self,
                 data_dir: pb.Path,
                 transform: DataTransform,
                 task: Literal['task_1', 'task_2'] = 'task_1',
                 split: Literal['train', 'eval', 'test'] = 'train',
                 max_conversation_len: Optional[int] = 24,
                 max_utterance_len: Optional[int] = 10,
                 filter_data: bool = False,
                 batch_size: int = 24) -> None:
        super(ECACDataset, self).__init__()

        # Argument Guards
        assert task in ('task_1', 'task_2'), 'invalid task type'
        assert split in ('train', 'eval', 'test'), 'invalid split type'
        assert max_utterance_len is None or max_utterance_len > 0, 'max_utterance_len must be greater than zero'
        assert max_conversation_len is None or max_conversation_len > 0, 'max_conversation_len must be greater than zero'
        if filter_data and split != 'train':
            raise ValueError('filter can only be used with the train split')

        # Data paths
        self.__video_pattern: str = r'dia\d+utt\d+\.mp4'
        self.__data_dir_path = data_dir / split / task
        self.__data_path = self.__data_dir_path / f'{task}_{split}.json'
        self.__video_dir_path = self.__data_dir_path / 'video'

        # Dataset settings
        self.task: Literal['task_1', 'task_2'] = task
        self.split: Literal['train', 'eval', 'test'] = split
        self.emotions: List[str] = ['neutral', 'anger', 'surprise', 'sadness', 'joy', 'disgust', 'fear']
        self.emotion_to_class = {e: i for i, e in enumerate(self.emotions)}
        self.class_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.num_emotions = len(self.emotions)

        # Transformation options
        self.__batch_size = batch_size
        self.__filter_data = filter_data
        self.__transform = transform
        self.__max_utterance_len = max_utterance_len
        self.__max_conversation_len = max_conversation_len

        # Integrity checks
        if not self.__data_dir_path.exists():
            raise FileNotFoundError(self.__data_dir_path)
        if not self.__data_path.exists():
            raise FileNotFoundError(self.__data_path)
        if self.task == 'task_2':
            if not self.__video_dir_path.exists():
                raise FileNotFoundError(self.__video_dir_path)
            if len(video_paths := list(self.__video_dir_path.iterdir())) == 0:
                raise Exception('no files were found under ' + str(self.__video_dir_path))
            if not all(map(lambda x: re.fullmatch(self.__video_pattern, x.name), video_paths)):
                raise Exception('found invalid files under ' + str(self.__video_dir_path))

        # Read and prepare the dataset
        self.data = self.__prepare_dataset()

    def __getitem__(self, index: int) -> EmotionCauseData:
        # Select the id of the conversation from the unique list of ids
        conversation_id = self.__conversation_ids[index]
        
        # Get metadata for conversations
        metadata_conversations = self.data.metadata.conversation[self.data.metadata.conversation['conversation_id'] == conversation_id]

        # Get metadata for labels
        if self.data.metadata.emotion_cause_pairs is not None:
            metadata_label = self.data.metadata.emotion_cause_pairs[self.data.metadata.emotion_cause_pairs['conversation_id'] == conversation_id]
        else:
            metadata_label = None

        # Aggregate metadata
        metadata = EmotionCauseMetaData(metadata_conversations, metadata_label)
        
        # For each feature select only the relevant one for the current conversation id
        encoding = tree_map(lambda x: x[index], self.data.encoding)

        # Aggregate metadata and encoding
        return EmotionCauseData(metadata, encoding)

    def __len__(self) -> int:
        return len(self.__conversation_ids)

    def __prepare_dataset(self) -> EmotionCauseData:
        raw_data = self.__read_dataset(self.__data_path)
        metadata = self.__tabulate_data(raw_data)
        metadata = self.__filter_dataset(metadata)
        encoding = self.__encode_data(metadata)
        return EmotionCauseData(metadata, encoding)

    def __filter_dataset(self, dataset: EmotionCauseMetaData) -> EmotionCauseMetaData:
        if not self.__filter_data or dataset.emotion_cause_pairs is None:
            return dataset
        if self.__max_conversation_len:
            # Get number of utterances for each conversation
            conversation_lengths = dataset.conversation['conversation_id'] \
                .value_counts(sort=False) \
                .rename('number_of_utterances') \
                .reset_index()

            # Get the conversations that have at most `max_conversation_len` utterances
            kept_conversation_ids = conversation_lengths[conversation_lengths['number_of_utterances'] <= self.__max_conversation_len]['conversation_id']

            # Remove the other from both data and labels
            dataset.conversation = dataset.conversation[dataset.conversation['conversation_id'].isin(kept_conversation_ids)] \
                .reset_index(drop=True)
            dataset.emotion_cause_pairs = dataset.emotion_cause_pairs[dataset.emotion_cause_pairs['conversation_id'].isin(kept_conversation_ids)] \
                .reset_index(drop=True)
        if self.__max_utterance_len:
            # Get number of units for each utterance
            utterance_lengths = dataset.conversation['text'] \
                .apply(lambda x: len(x.split(' '))) \
                .rename('number_of_units') \
                .reset_index()
            utterance_lengths['conversation_id'] = dataset.conversation['conversation_id']

            # Take only the conversations with at most `max_utterance_len` number of units
            skip_conversation_ids = utterance_lengths[utterance_lengths['number_of_units'] >= self.__max_utterance_len]['conversation_id']

            # Remove the other from both data and labels
            dataset.conversation = dataset.conversation[~dataset.conversation['conversation_id'].isin(skip_conversation_ids)] \
                .reset_index(drop=True)
            dataset.emotion_cause_pairs = dataset.emotion_cause_pairs[~dataset.emotion_cause_pairs['conversation_id'].isin(skip_conversation_ids)] \
                .reset_index(drop=True)
        return dataset

    def __read_dataset(self, path: Path) -> List[EmotionCauseConversation]:
        with open(path, 'r') as dataset_file:
            return json.load(dataset_file)

    def __tabulate_data(self, dataset: List[EmotionCauseConversation]) -> EmotionCauseMetaData:
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
                else:
                    data_[-1]['emotion'] = None

                # Obtain video_path 
                if self.task == 'task_2':
                    video_name = f'dia{conversation["conversation_ID"]}utt{utterance["utterance_ID"]}.mp4'
                    video_path = self.__video_dir_path / video_name
                    assert os.path.exists(video_path), f'no video file was found on disk at {video_path}'
                    data_[-1]['video_path'] = str(video_path)
                else:
                    data_[-1]['video_path'] = None

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
                else:
                    labels_[-1]['cause_start'] = None
                    labels_[-1]['cause_stop'] = None

        # Aggregate data and labels
        data: DataFrame = pd.DataFrame(data_)
        labels: DataFrame | None = pd.DataFrame(labels_) if self.split == 'train' else None
        return EmotionCauseMetaData(data, labels)

    def __encode_data(self, dataset: EmotionCauseMetaData) -> EmotionCauseEncoding:
        # Accumulate all batches
        batches = []

        # Find all unique conversations
        self.__conversation_ids = dataset.conversation['conversation_id'].unique()

        # Split the transformation into batches
        n_batches: int = math.ceil(len(self.__conversation_ids) / self.__batch_size)

        # Encode each batch of conversations
        for batch_index in tqdm.trange(n_batches, desc='batch'):
            batch_start: int = batch_index * self.__batch_size
            batch_stop: int = min(batch_start + self.__batch_size, len(self.__conversation_ids))
            batch_ids: np.ndarray = self.__conversation_ids[batch_start: batch_stop]

            # Select batch of conversations
            batch_data = dataset.conversation[dataset.conversation['conversation_id'].isin(batch_ids)]

            # Select batch of labels
            if dataset.emotion_cause_pairs is not None:
                batch_labels = dataset.emotion_cause_pairs[dataset.emotion_cause_pairs['conversation_id'].isin(batch_ids)]
            else:
                batch_labels = None

            # Encode the samples
            batches.append(dict(self.__transform((batch_data, batch_labels))))

        # Flatten batches
        data = {}
        for key in batches[0].keys():
            data[key] = jnp.concatenate([batches[idx][key] for idx in range(len(batches))], axis=0)
        return cast(EmotionCauseEncoding, data)
