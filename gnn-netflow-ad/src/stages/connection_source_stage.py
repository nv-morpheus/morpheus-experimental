# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#


import logging
import pathlib
import typing
import time

import mrc

# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import MessageMeta as CppMessageMeta
from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import ControlMessage
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

import cudf
import cupy as cp

logger = logging.getLogger(__name__)


@register_stage("connection-source", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class ConnectionSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.
    Source stage is used to simulate polling Athena and emitting data into the pipeline.
    """

    def __init__(self,
                 c: Config, df, repeat_count=4, events_per_loop=10_000):
        super().__init__(c)

        self._df = df
        self._repeat_count = repeat_count
        self._events_per_loop = events_per_loop

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "connection-source"

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node"""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        node = builder.make_source(self.unique_name, self._generate_frames())

        return node

    def _generate_frames(self) -> typing.Iterable[ControlMessage]:

        df = self._df.copy()

        for i in range(self._repeat_count):

            df = df.sample(n=self._events_per_loop, replace=True)
            msg_meta = MessageMeta(df)
            ctrl_msg = ControlMessage()
            ctrl_msg.payload(msg_meta)

            # If we are looping, copy the object. Do this before we push the object in case it changes
            if (i + 1 < self._repeat_count):
                df = df.copy()

                # Shift the index to allow for unique indices without reading more data
                df.index += len(df)

            yield ctrl_msg
