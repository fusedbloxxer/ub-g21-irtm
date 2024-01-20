from typing import Any, TypedDict, Optional, List, Literal, Dict, Tuple, cast
from torch.utils.data import Dataset
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
import pathlib as pb
import json
import re

from .types import Emotion, SubTask, DataSplit
from .types import EmotionCauseConversation
from .types import ECACData


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
    def emotion2label(self) -> Dict[str, int]:
        return self.__emotion2label

    @property
    def label2emotion(self) -> Dict[int, str]:
        return self.__label2emotion

    def __getitem__(self, index: int) -> EmotionCauseConversation:
        return self.__data[index]

    def __len__(self) -> int:
        return len(self.__data)

    def __read_data(self):
        with open(self.__metadata_path, 'r') as dataset_file:
            self.__data: ECACData = json.load(dataset_file)
