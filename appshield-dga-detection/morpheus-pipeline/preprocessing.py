# Copyright (c) 2022, NVIDIA CORPORATION.
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

import dataclasses
import logging
import typing
from functools import partial

import cudf
import cupy as cp
import morpheus._lib.stages as srfs
import srf
from morpheus.config import Config
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.multi_inference_message import (
    InferenceMemory, MultiInferenceMessage, MultiInferenceNLPMessage,
    get_input, set_input)
from morpheus.messages.multi_message import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import \
    PreprocessBaseStage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferenceMemoryDGA(InferenceMemory):
    input: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __post_init__(self, input, seq_ids):
        self.input = input
        self.seq_ids = seq_ids


class PreprocessingDGAStage(PreprocessBaseStage):
    def __init__(self, c: Config, feature_columns: typing.List[str]):
        super().__init__(c)
        self._feature_columns = feature_columns
        self._features_len = len(self._feature_columns)

    @property
    def name(self) -> str:
        return "preprocess-dga"

    def supports_cpp_node(self):
        return False

    def _pre_process_batch(self, x: MultiMessage) -> MultiInferenceNLPMessage:
        df = x.get_meta()
        df = cudf.from_pandas(df)

        input_df = df[self._feature_columns]
        for col in input_df.columns:
            input_df[col] = input_df[col].astype("float32")

        data = cp.asarray(input_df.to_cupy())
        count = input_df.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = len(self._feature_columns) - 1

        memory = InferenceMemoryDGA(count=count, input=data, seq_ids=seg_ids)
        infer_message = MultiInferenceMessage(
            meta=x.meta,
            mess_offset=x.mess_offset,
            mess_count=x.mess_count,
            memory=memory,
            offset=0,
            count=memory.count,
        )
        return infer_message

    def _get_preprocess_fn(
        self,
    ) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:

        pre_process_batch_fn = self._pre_process_batch

        return partial(pre_process_batch_fn)

    def _get_preprocess_node(self, seg: srf.Builder):
        return srfs.PreprocessingRWStage(seg, self.unique_name)
