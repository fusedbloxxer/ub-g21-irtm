import typing as t
from typing import Any, cast
from jax import Array, vmap, jit
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from dataclasses import dataclass
from transformers import BatchEncoding, PreTrainedTokenizerFast
from datasets.formatting.formatting import LazyBatch
from functools import partial


@dataclass
class DataTokenize(object):
    tokenizer: PreTrainedTokenizerFast
    max_length: t.Optional[int] = 93
    padding: t.Literal['longest', 'max_length'] = 'max_length'

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
            return_tensors='jax',
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_offsets_mapping=True
        )


@dataclass
class DataTransform(object):
    tokenize: t.Callable[[t.List[str]], BatchEncoding]
    max_length: t.Optional[int] = 33

    def __call__(self, x: t.Tuple[DataFrame, DataFrame | None]) -> BatchEncoding:
        assert len(x) > 0, 'at least one element must be provided'
        conv_ids = jnp.asarray(x[0]['conversation_id'].unique())
        has_labels: bool = x[1] is not None
        batch_size: int = len(conv_ids)

        # Separate data
        X_df = x[0]
        y_df = cast(DataFrame, x[1])
        has_video: bool = len(X_df) > 0 and not X_df['video_path'].isnull().any()

        # Where to split the batch to obtain each individual conversation
        conv_lengths = jnp.asarray(X_df['conversation_id'].value_counts(sort=False).to_numpy())
        conv_splits: Array = conv_lengths.cumsum()

        # Tokenize all texts at once
        X: BatchEncoding = self.tokenize(X_df['text'].tolist())
        seq_len: int = X.data['input_ids'].shape[1]

        # Shape is (C, T)
        # C - is a varying sum of the lengths of all conversation in the batch
        # T - is the sequence length which is fixed by the tokenize function
        # We need to ensure that conversations are aligned and create a batch dimension
        # To do so we can pad with zeros all conversations s.t. that they become aligned
        pad_index = jnp.repeat(conv_splits, self.max_length - conv_lengths)

        # Pad and reshape the `input_ids` to be (B, C, T, ...)
        X['input_ids'] = jnp.insert(X.data['input_ids'], pad_index, jnp.zeros((seq_len,)), axis=0)
        X['input_ids'] = jnp.reshape(X.data['input_ids'], (batch_size, -1, seq_len))
        conv_len = X.data['input_ids'].shape[1]

        # Pad and reshape the `attention_mask`
        X['attention_mask'] = jnp.insert(X.data['attention_mask'], pad_index, jnp.zeros((seq_len,)), axis=0)
        X['attention_mask'] = jnp.reshape(X.data['attention_mask'], (batch_size, -1, seq_len))

        # Pad and reshape the `offset_mapping`
        X['offset_mapping'] = jnp.insert(X.data['offset_mapping'], pad_index, jnp.zeros((2,)), axis=0)
        X['offset_mapping'] = jnp.reshape(X.data['offset_mapping'], (batch_size, -1, seq_len, 2))

        # Create a mask to indicate which utterances represent real data and not just padding
        num_and_pad = jnp.column_stack((conv_lengths, self.max_length - conv_lengths)).reshape(-1)
        X['input_mask'] = jnp.tile(jnp.array([1, 0]), jnp.array([batch_size,]))
        X['input_mask'] = jnp.repeat(X.data['input_mask'], num_and_pad, axis=0).reshape((batch_size, -1))

        if has_labels:
            has_cause: bool = len(y_df) > 0
            has_cause_span: bool = has_cause and not cast(bool, y_df['cause_start'].isnull().any())

            # Provide emotions for all utterances
            X['emotion_labels'] = jnp.asarray(X_df['emotion'].to_numpy())
            X['emotion_labels'] = jnp.insert(X.data['emotion_labels'], pad_index, 0, axis=0)
            X['emotion_labels'] = jnp.reshape(X.data['emotion_labels'], (batch_size, -1))

            # For a mapping of type { CONV_ID: UTTERANCE_ID } for each element in the batch 
            b_conv_ids = jnp.repeat(conv_ids, conv_len)
            b_utter_ids = jnp.tile(jnp.arange(conv_len), (batch_size,))
            b_conv_utter_ids = map(tuple, jnp.column_stack((b_conv_ids, b_utter_ids)).tolist())

            # For each utterance in the batch check if it's a cause
            conv_cause_pairs: t.Set[t.Tuple[int, ...]] = {tuple(x) for x in y_df[['conversation_id', 'cause_id']].values.tolist()}
            X['cause_labels'] = jnp.asarray([x in conv_cause_pairs for x in b_conv_utter_ids], dtype=jnp.int32).reshape(batch_size, -1)

            # Create causality matrices for the text ranges
            # Where each element can represent an index from 0 to T  
            # (B, 2, C, C): where B is batch_size and 2 stands for start/stop token_index

            # Allocate the matrix and treat zero as the `not_found` token_index
            X['cause_span'] = np.zeros((batch_size, 2, conv_len, conv_len), dtype=jnp.int32)

            if has_cause_span:
                # Map (conversation_id, cause_id) to the original batch_encoding
                conv_to_offset = jnp.insert(conv_lengths[:-1], obj=0, values=0).cumsum()
                conv_to_offset = {idx.item(): offset.item() for idx, offset in zip(conv_ids, conv_to_offset)}
                b_cause_ids = jnp.asarray((y_df['cause_id'] + y_df['conversation_id'].map(conv_to_offset)).to_numpy())
                b_cause_idx = jnp.column_stack((b_cause_ids, y_df['cause_start'].to_numpy(), y_df['cause_stop'].to_numpy()))

                # Extract token_index using char_to_token
                char_to_token = cast(Any, partial(DataTransform.char_to_token, X))
                token_id_span = np.apply_along_axis(char_to_token, 1, b_cause_idx)
                token_id_span = jnp.concatenate((y_df['conversation_id'].to_numpy()[..., None], b_cause_idx, token_id_span), axis=1)

                # Map (conversation_id, utterance_id) and (conversation_id, cause_id) to batch
                batch_ids = jnp.arange(batch_size)
                conv_to_batch = {c.item(): b.item() for c, b in zip(conv_ids, batch_ids)}
                token_id_span = jnp.concatenate((np.vectorize(conv_to_batch.__getitem__)(token_id_span[:, 0])[..., None], token_id_span), axis=1)

                # 5. Add them to X['cause_span][:, 0, utter, cause] and X['cause_span][:, 0, utter, cause]
                for i in range(token_id_span.shape[0]):
                    # Retrieve token range
                    start_token_index: int = token_id_span[i, -2].item()
                    stop_token_index: int = token_id_span[i, -1].item()

                    # Select position in matrix to insert to
                    batch_index: int = token_id_span[i, 0].item()
                    utterance_index: int = y_df.iloc[i]['utterance_id']
                    cause_index: int = y_df.iloc[i]['cause_id']

                    # Mark
                    X.data['cause_span'][batch_index, 0, utterance_index, cause_index] = start_token_index
                    X.data['cause_span'][batch_index, 1, utterance_index, cause_index] = stop_token_index

            # Create mask to indicate which indices to focus at
            X['cause_mask'] = jnp.astype(X['cause_span'] != 0, jnp.float32)

        # Return an aggregation of the data and the labels
        return X

    @staticmethod
    def char_to_token(text_encoding: BatchEncoding, cause: Array) -> Array:
        # Obtain the indices for the token_ids at both ends of the range
        start_id = text_encoding.char_to_token(cause[0].item(), cause[1].item())
        stop_id  = text_encoding.char_to_token(cause[0].item(), cause[2].item())
        return jnp.array([start_id, stop_id])
