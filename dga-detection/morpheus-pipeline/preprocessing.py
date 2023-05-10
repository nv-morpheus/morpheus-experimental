# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from functools import partial

import cupy as cp
import mrc
from messages import InferenceMemoryDGA
from messages import MultiInferenceDGAMessage

import cudf

from morpheus.config import Config
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage


class PreprocessDGAStage(PreprocessBaseStage):
    """
    Prepare NLP input DataFrames for inference.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    add_special_tokens : bool
        Whether or not to encode the sequences with the special tokens of the BERT classification model.
    stride : int
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.
    column : str
        Name of the column containing the data that needs to be preprocessed.

    """

    def __init__(self, c: Config, column: str = "data", truncate_length: int = 100):
        super().__init__(c)

        self._column = column
        self._truncate_length = truncate_length
        self._fea_length = c.feature_length

    @property
    def name(self) -> str:
        return "preprocess-dga"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, column: str, truncate_len: int) -> MultiInferenceNLPMessage:

        df = x.get_meta()[[column]]
        df[column] = df[column].str.slice_replace(truncate_len, repl='')

        split_ser = df[column].str.findall(r"[\w\W\d\D\s\S]")
        split_df = split_ser.to_frame()
        split_df = cudf.DataFrame(split_df[column].to_arrow().to_pylist())
        columns_cnt = len(split_df.columns)

        # Replace null's with ^.
        split_df = split_df.fillna("^")
        temp_df = cudf.DataFrame()
        for col in range(0, columns_cnt):
            temp_df[col] = split_df[col].str.code_points()
        del split_df

        # Replace ^ ascii value 94 with 0.
        temp_df = temp_df.replace(94, 0)
        temp_df.index = df.index
        # temp_df["len"] = df["len"]
        if "type" in df.columns:
            temp_df["type"] = df["type"]
        temp_df[column] = df[column]

        temp_df = temp_df.drop("domain", axis=1)

        domains = cp.asarray(temp_df.to_cupy()).astype("long")

        input = cp.zeros((domains.shape[0], fea_len))
        input[:domains.shape[0], :domains.shape[1]] = domains
        input = input.astype("long")

        count = input.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(x.mess_offset, x.mess_offset + count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryDGA(count=input.shape[0], domains=input, seq_ids=seg_ids)

        infer_message = MultiInferenceDGAMessage.from_message(x, memory=memory)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:

        return partial(PreprocessDGAStage.pre_process_batch,
                       fea_len=self._fea_length,
                       column=self._column,
                       truncate_len=self._truncate_length)

    def _get_preprocess_node(self, builder: mrc.Builder):
        raise NotImplementedError("C++ node not implemented")
