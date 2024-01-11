import typing as t
from torch.utils.data import Dataset
from dataclasses import dataclass
from functools import partial
import pathlib as pb
import pandas as pd
import json
import os
import re


Emotion: t.TypeAlias = t.Literal['neutral', 'anger', 'surprise', 'sadness', 'joy', 'disgust', 'fear']


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
    conversation: t.List[Utterance]
    emotion_cause: t.List[t.List[str]]

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

    @staticmethod
    def to_df(conversations: t.List['Conversation'],
              emotion_to_class: t.Dict[str, int],
              with_text_range: bool,
              with_pairs: bool) -> t.Tuple[pd.DataFrame, pd.DataFrame | None]:
        # Transform to pandas format for faster processing
        labels_: t.List[t.Dict[str, t.Any]] = []
        data_: t.List[t.Dict[str, t.Any]] = []

        # Aggregate all conversations in two separate DFs
        for sample in conversations:
            for utterance in sample.conversation:
                # Obtain emotion
                label_or_none: int | None = emotion_to_class.get(str(utterance.emotion), None)

                # Add entry with ids starting from zero
                data_.append({
                    'conversation_id': sample.conversation_id - 1,
                    'utterance_id': utterance.uterrance_id - 1,
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'emotion': label_or_none,
                    'video_path': utterance.video_path,
                })

            if with_pairs:
                for [emotion, cause] in sample.emotion_cause:
                    # Obtain emotion
                    utterance_id: int = int(emotion.split('_')[0]) - 1
                    emotion_str: str = ''.join(emotion.split('_')[1:])
                    label: int = emotion_to_class[emotion_str]

                    # Obtain cause
                    cause_id: int = int(cause.split('_')[0]) - 1

                    # Find the indices of the cause for task 1 only
                    if with_text_range:
                        cause_str: str = ''.join(cause.split('_')[1:])
                        cause_start = sample.conversation[cause_id].text.find(cause_str)
                        cause_stop = cause_start + len(cause_str) - 1
                        assert cause_start != -1, 'could not find cause range (conv_id: {}, utt_id: {}, cause_id: {}, cause_str: {})' \
                            .format(sample.conversation_id, utterance_id, cause_id, cause_str)
                    else:
                        cause_start = None
                        cause_stop = None

                    # Add entry with ids starting from zero
                    labels_.append({
                        'conversation_id': sample.conversation_id - 1,
                        'utterance_id': utterance_id,
                        'emotion': label,
                        'cause_id': cause_id,
                        'cause_start': cause_start,
                        'cause_stop': cause_stop, 
                    })

        labels = pd.DataFrame(labels_) if with_pairs else None
        data = pd.DataFrame(data_)
        return data, labels


class ECACDataset(Dataset):
    def __init__(self,
                 data_dir: pb.Path,
                 task: t.Literal['task_1', 'task_2'] = 'task_1',
                 split: t.Literal['train', 'eval', 'test'] = 'train') -> None:
        super().__init__()
        assert task in ('task_1', 'task_2'), 'invalid task type'
        assert split in ('train', 'eval', 'test'), 'invalid split type'

        # Data paths
        self.data_dir_path = data_dir / 'competition' / split / task
        self.metadata_path = self.data_dir_path / f'{task}_{split}.json'
        self.video_dir_path = self.data_dir_path / 'video'
        self.video_pattern: str = r'dia\d+utt\d+\.mp4'

        # Dataset config
        self.task: t.Literal['task_1', 'task_2'] = task
        self.split: t.Literal['train', 'eval', 'test'] = split
        self.emotions = ['neutral', 'anger', 'surprise', 'sadness', 'joy', 'disgust', 'fear']
        self.emotion_to_class = {e: i for i, e in enumerate(self.emotions)}
        self.class_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.num_emotions = len(self.emotions)

        # Integrity checks
        if not self.data_dir_path.exists():
            raise FileNotFoundError(self.data_dir_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        if self.task == 'task_2':
            if not self.video_dir_path.exists():
                raise FileNotFoundError(self.video_dir_path)
            if len(video_paths := list(self.video_dir_path.iterdir())) == 0:
                raise Exception('no files were found under ' + str(self.video_dir_path))
            if not all(map(lambda x: re.fullmatch(self.video_pattern, x.name), video_paths)):
                raise Exception('found invalid files under ' + str(self.video_dir_path))

        # Read metadata in-memory
        with open(self.metadata_path, 'r') as file:
            # Json -> Conversation DataClass
            make_conversation = partial(Conversation.from_dict, video_dir=None if self.task == 'task_1' else self.video_dir_path)
            metadata = list(map(make_conversation, json.load(file)))

            # Conversation DataClass -> DataFrame
            self.data_, self.labels_ = Conversation.to_df(
                emotion_to_class=self.emotion_to_class,
                with_text_range=self.task == 'task_1',
                with_pairs=self.split == 'train',
                conversations=metadata,
            )

    def __getitem__(self, index: int):
        data = self.data_[self.data_['conversation_id'] == index]

        if self.split == 'train':
            assert self.labels_ is not None, 'no labels were found for train split'
            labels = self.labels_[self.labels_['conversation_id'] == index]
        else:
            labels = None

        return (data, labels)

    def __len__(self) -> int:
        return len(self.data_['conversation_id'].unique())
