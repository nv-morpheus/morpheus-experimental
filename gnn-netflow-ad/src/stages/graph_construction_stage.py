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

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

from src.preprocess import NetflowPreprocessor
import cudf
import cupy as cp


@register_stage("graph-construction", modes=[PipelineModes.OTHER])
class GraphConstructionStage(SinglePortStage):

    def __init__(self, config: Config, scaler_path: str,
                 edge_columns=['IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION_MILLISECONDS'], node_dim=32,
                 src_ip='IPV4_SRC_ADDR', dst_ip='IPV4_DST_ADDR'):
        """
        Create a fraud-graph-construction stage

        Parameters
        ----------
        c : Config
            The Morpheus config object
        scaler_path: str
            Path to the Scaler object to normalize edge features.
        edge_columns: [str]
            List of features in an input dataframe to consider as edge features.
        node_dim: int
            Dimension of computed node embeddings.
        src_ip: str
            Column name to find source IP address in.
        dst_ip: str
            Column name to fine destination IP address in.
        """
        super().__init__(config)

        # Instantiate the preprocesser
        self._processor = NetflowPreprocessor(
            cudf.DataFrame(),
            edge_columns=edge_columns,
            node_dim=node_dim,
            normalize=False,
            src_ip=src_ip,
            dst_ip=dst_ip
        )

        # Load the edge scaler
        self._processor.load_scaler(path=scaler_path)

    @property
    def name(self) -> str:
        return "graph-construction"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def _process_message(self, message: ControlMessage) -> ControlMessage:
        graph_data, ip_map = self._processor.process_single(message.payload().copy_dataframe())
        message.set_metadata("graph", graph_data)
        message.set_metadata("ip_map", ip_map)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_node, node)
        return node