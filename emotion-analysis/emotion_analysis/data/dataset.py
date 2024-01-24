from typing import Any, TypedDict, Optional, List, Literal, Dict, Tuple, cast, Sequence
from torch.utils.data import Dataset
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
import pathlib as pb
import numpy as np
import json
import re

from .types import Emotion, SubTask, DataSplit
from .types import EmotionCauseConversation
from .types import ECACData, DataStats


class ECACDataset(Dataset):
    def __init__(
        self,
        data_dir: pb.Path,
        *,
        split: DataSplit,
        subtask: SubTask,
    ) -> None:
        super(ECACDataset, self).__init__()

        # Argument Guards
        assert subtask in ('1', '2'), 'invalid subtask type'
        assert split in ('train', 'trial', 'test'), 'invalid split type'

        # Data paths
        self.__data_dir_path: Path = data_dir / split
        self.__subtask_path: Path = self.__data_dir_path / f'subtask_{subtask}'
        self.__metadata_path = self.__subtask_path / f'{split}.json'
        self.__video_dir_path = self.__subtask_path / 'video'
        self.__video_pattern: str = r'dia\d+utt\d+\.mp4'

        # Dataset settings
        self.__split: DataSplit = split
        self.__subtask: SubTask = subtask
        self.__emotions: List[Emotion] = ['neutral', 'anger', 'surprise', 'sadness', 'joy', 'disgust', 'fear']
        self.__emotion2label: Dict[str ,int] = {e: i for i, e in enumerate(self.__emotions)}
        self.__label2emotion: Dict[int, str] = {i: e for i, e in enumerate(self.__emotions)}

        # Integrity checks
        if not self.__data_dir_path.exists():
            raise FileNotFoundError(self.__data_dir_path)
        if not self.__metadata_path.exists():
            raise FileNotFoundError(self.__metadata_path)
        if self.__subtask == 'task_2':
            if not self.__video_dir_path.exists():
                raise FileNotFoundError(self.__video_dir_path)
            if len(video_paths := list(self.__video_dir_path.iterdir())) == 0:
                raise Exception('no files were found under ' + str(self.__video_dir_path))
            if not all(map(lambda x: re.fullmatch(self.__video_pattern, x.name), video_paths)):
                raise Exception('found invalid files under ' + str(self.__video_dir_path))

        # Read the dataset
        self.__read_data()

    @property
    def subtask(self) -> SubTask:
        return self.__subtask

    @property
    def split(self) -> DataSplit:
        return self.__split

    @property 
    def emotions(self) -> List[Emotion]:
        return self.__emotions

    @property
    def num_emotions(self) -> int:
        return len(self.emotions)

    @property
    def emotion2label(self) -> Dict[str, int]:
        return self.__emotion2label

    @property
    def label2emotion(self) -> Dict[int, str]:
        return self.__label2emotion

    @property
    def stats(self) -> DataStats:
        return self._stats

    @property
    def emotion_labels(self) -> Sequence[int]:
        return self._stats['emotion_labels']

    @property
    def samples_per_emotion(self) -> np.ndarray:
        return np.bincount(self.emotion_labels)

    def __getitem__(self, index: int) -> EmotionCauseConversation:
        sample = self.__data[index]

        return sample

    def __len__(self) -> int:
        return len(self.__data)

    def __read_data(self):
        # Read raw dataset
        with open(self.__metadata_path, 'r') as dataset_file:
            self.__data: ECACData = json.load(dataset_file)

        # Track Statistics
        self._stats: DataStats = {
            'emotion_labels': [],
        }

        # Inject missing values
        for sample in self.__data:
            # Some samples have missing values and the data type cannot be determined
            sample['has_spans'] = self.split == 'train' and self.subtask == '1'
            sample['has_emotions'] = self.split == 'train'
            sample['has_causes'] = self.split == 'train'
            sample['has_video'] = self.subtask == '2'

            for utterance in sample['conversation']:
                # Inject missing video name and path
                if self.subtask == '2':
                    video_name_fallback: str = f"dia{sample['conversation_ID']}utt{utterance['utterance_ID']}.mp4"
                    video_name = utterance.get('video_name') or video_name_fallback
                    utterance['video_path'] = self.__video_dir_path / video_name
                    utterance['video_name'] = video_name

                # Skip non-train examples
                if self.split != 'train':
                    continue

                if 'emotion' in utterance:
                    # Inject emotion as label
                    utterance['emotion_label'] = self.emotion2label[utterance['emotion']]

                    # Increment for each sample found
                    self._stats['emotion_labels'].append(utterance['emotion_label'])

    @classmethod
    def read_data(cls, data_dir: pb.Path, subtask: SubTask) -> Dict[DataSplit, 'ECACDataset']:
        ds_train = cls(data_dir, split='train', subtask=subtask)
        ds_trial = cls(data_dir, split='trial', subtask=subtask)
        ds_test  = cls(data_dir,  split='test', subtask=subtask)
        return { 'train': ds_train, 'trial': ds_trial, 'test': ds_test }

