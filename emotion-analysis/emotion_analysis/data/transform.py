import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Any, Callable, Generic, TypeVar, cast

import flax.core.frozen_dict as frozen_dict
import jax.numpy as jnp
import numpy as np
from datasets.formatting.formatting import LazyBatch
from jax import Array, jit, vmap
from pandas import DataFrame
from transformers import BatchEncoding, PreTrainedTokenizerFast

from .dataset import ECACDataset
from ..utils.weight import ClassWeight
from .types import EmotionCauseConversation, EmotionCauseEncoding

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')


@dataclass
class Transform(Generic[T_in, T_out], ABC):
    @abstractmethod
    def __call__(self, input_: T_in) -> T_out:
        raise NotImplementedError()

    @staticmethod
    def chain(*transforms: 'Transform') -> 'Transform[T_in, T_out]':
        def apply_chain(data):
            for transform in transforms:
                data = transform(data)
            return data
        return cast(Any, apply_chain)


@dataclass
class TokenizeTransform(Transform[str | t.List[str] | t.Dict[str, t.Any] | LazyBatch, BatchEncoding]):
    tokenizer: PreTrainedTokenizerFast
    padding: t.Literal['longest', 'max_length'] = 'max_length'
    max_seq_len: t.Optional[int] = field(kw_only=True, default=93)

    def __call__(self, sample: str | t.List[str] | t.Dict[str, t.Any] | LazyBatch) -> BatchEncoding:
        # Extract `text` from sample
        if   isinstance(sample, (str, list)):
            batch: t.List[str] | str = sample
        elif isinstance(sample, (dict, LazyBatch)):
            batch: t.List[str] | str = cast(Any, sample['text'])
        else:
            raise ValueError('cannot tokenize, invalid type for sample: {}'.format(type(sample)))

        # Tokenize entry or batch
        return self.tokenizer(
            text=batch,
            truncation=True,
            padding=self.padding,
            return_tensors='np',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_seq_len,
        )


@dataclass
class EncodeTransform(Transform[EmotionCauseConversation, EmotionCauseEncoding]):
    tokenize: t.Callable[[t.List[str]], BatchEncoding]
    max_conv_len: int = field(kw_only=True, default=33)

    def __call__(self, sample: EmotionCauseConversation) -> EmotionCauseEncoding:
        emotion_causes = defaultdict(list)
        cause_spans = defaultdict(list)
        emotion_labels = []
        data = {}

        ### Perform tokenization ###
        encoding = self.tokenize([utterance['text'] for utterance in sample['conversation']])
        conv_len = encoding.data['input_ids'].shape[0]
        pad_len = self.max_conv_len - conv_len

        # Add padding to the conversation
        input_ids = np.pad(encoding.data['input_ids'], ((0, pad_len), (0, 0)))
        data['input_ids'] = jnp.asarray(input_ids)
        attn_mask = np.pad(encoding.data['attention_mask'], ((0, pad_len), (0, 0)))
        data['uttr_attn_mask'] = jnp.asarray(attn_mask)

        # Create input mask to distinguish data from padding
        input_mask = np.zeros(self.max_conv_len, dtype=np.int32)
        input_mask[:len(sample['conversation'])] = 1
        data['conv_attn_mask'] = jnp.asarray(input_mask)

        ### Perform label extraction ### 
        if sample['has_emotions']:
            for utterance in sample['conversation']:
                # Extract emotions as labels
                emotion_labels.append(utterance.get('emotion_label', 0))

            # Add padding for alignment
            emotion_labels = np.pad(emotion_labels, (0, self.max_conv_len - len(emotion_labels)))
            data['emotion_labels'] = emotion_labels

        if sample['has_causes']:
            for emotion_cause in sample.get('emotion-cause_pairs', []):
                # Split emotion and cause
                emotion, cause = emotion_cause

                # Extract utterance 
                utterance = int(emotion.split('_')[0]) - 1

                # Extract cause
                cause_info = cause.split('_')
                cause = int(cause_info[0]) - 1
                emotion_causes[utterance].append(cause)

                # Extract span char index
                if sample['has_spans']:
                    span = cause_info[1]
                    source = sample['conversation'][cause]['text']
                    boc = source.find(span)
                    eoc = boc + len(span) - 1
                    cause_spans[utterance].append((boc, eoc))

            # Add padding for alignment
            causes = list(chain.from_iterable(emotion_causes.values()))
            cause_labels = np.zeros(self.max_conv_len, dtype=np.int64)
            cause_labels[causes] = 1
            data['cause_labels'] = jnp.asarray(cause_labels)

        if sample['has_spans']:
            boc_matrix = np.zeros((self.max_conv_len, self.max_conv_len), dtype=np.int64)
            eoc_matrix = np.zeros((self.max_conv_len, self.max_conv_len), dtype=np.int64)

            # Fill in ExC with token indices
            for utterance in cause_spans:
                for i, (char_boc, char_eoc) in enumerate(cause_spans[utterance]):
                    cause = emotion_causes[utterance][i]
                    boc_index = encoding.char_to_token(cause, char_boc)
                    eoc_index = encoding.char_to_token(cause, char_eoc)
                    boc_matrix[utterance][cause] = boc_index
                    eoc_matrix[utterance][cause] = eoc_index

            # Concatenate them
            span_matrix = np.stack((boc_matrix, eoc_matrix), axis=-1)
            data['span_mask'] = jnp.asarray(span_matrix.sum(axis=-1) != 0, np.int32) # todo attention to this! maybe it starts at zero => issue
            data['cause_span'] = jnp.asarray(span_matrix, np.int32)
        return cast(EmotionCauseEncoding, data)


@dataclass
class WeightTransform(Transform[EmotionCauseEncoding, EmotionCauseEncoding]):
    class_weight: ClassWeight

    def __call__(self, input_: EmotionCauseEncoding) -> EmotionCauseEncoding:
        if 'emotion_labels' in input_:
            input_['emotion_weight'] = jnp.asarray(self.class_weight(input_['emotion_labels']))
        return input_


@dataclass
class CollateTransform(Transform[t.List[EmotionCauseConversation], EmotionCauseEncoding]):
    transform: Transform[EmotionCauseConversation, EmotionCauseEncoding]

    def __call__(self, samples: t.List[EmotionCauseConversation]) -> EmotionCauseEncoding:
        assert len(samples) >= 1, 'the passed-in argument must contain at least one sample'
        batch: EmotionCauseEncoding = cast(Any, defaultdict(list))

        # Transform all samples and collect them
        for encoding in map(self.transform, samples):
            for key in encoding.keys():
                batch[key].append(encoding[key])

        # Aggregate them under a batch dimension
        for key in batch.keys():
            batch[key] = jnp.stack(batch[key], axis=0)

        return cast(EmotionCauseEncoding, frozen_dict.freeze(batch))
