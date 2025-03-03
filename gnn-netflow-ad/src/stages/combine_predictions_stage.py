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


import pathlib
import typing

import mrc
import torch
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

import cudf
import cupy as cp
from src.scripts import combine_preds


@register_stage("result-combine", modes=[PipelineModes.OTHER])
class CombinePredictionsStage(SinglePortStage):

    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "result_combine"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> MessageMeta:
        df = message.payload()
        ip_map = message.get_metadata("ip_map")
        graph = message.get_metadata("graph")
        preds = message.get_metadata("predictions")

        df_preds = combine_preds(
            cp.from_dlpack(graph.edge_index),
            ip_map,
            preds,
            df.copy_dataframe()
        )

        msg_meta = MessageMeta(df_preds)

        return msg_meta

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node