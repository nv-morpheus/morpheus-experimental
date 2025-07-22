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


import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

from src.guided_gae_model import GATEncoderWithEdgeAttr, GlobalEdgeEmbedding, DecoderWithGlobalEdge, GAEWithGlobalEdge
import torch
import torch.nn as nn
import cupy as cp
import cudf
from torch.cuda.amp import autocast


@register_stage("gnn-inference", modes=[PipelineModes.OTHER])
class GraphInferenceStage(SinglePortStage):

    def __init__(self,
                 config: Config,
                 model_file: str,
                 in_channels=32,
                 hidden_channels: int = 256,
                 edge_attr_dim: int = 3,
                 global_emb_dim: int = 128
                 ):
        super().__init__(config)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoder = GATEncoderWithEdgeAttr(in_channels, hidden_channels, edge_attr_dim)
        global_edge_embedding = GlobalEdgeEmbedding(edge_attr_dim, global_emb_dim)
        decoder = DecoderWithGlobalEdge(hidden_channels, edge_attr_dim, global_emb_dim)
        self._model = GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(self._device)

        self._model.load_state_dict(
            torch.load(model_file, weights_only=True)
        )

    @property
    def name(self) -> str:
        return "gnn-inference"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> ControlMessage:
        data = message.get_metadata("graph")
        with torch.no_grad():
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long).to(data.x.device)
            data = data.to(self._device)
            z = self._model.encode(data.x, data.edge_index, data.edge_attr[:, :-1])
            pos_edge_index = data.edge_index
            pos_pred = self._model.decode(z, pos_edge_index, data.edge_attr[:, :-1], data.batch)

        message.set_metadata("predictions", 1 - cp.from_dlpack(pos_pred.float()))  # Set predictions as a cupy array
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node