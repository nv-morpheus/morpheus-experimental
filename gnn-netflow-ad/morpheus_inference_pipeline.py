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
import os

import click

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging, reset_logging

import cudf

from src.stages.connection_source_stage import ConnectionSourceStage
from src.stages.graph_construction_stage import GraphConstructionStage
from src.stages.graph_inference_stage import GraphInferenceStage
from src.stages.combine_predictions_stage import CombinePredictionsStage

if __name__ == "__main__":
    test_data = cudf.read_parquet("artifacts/sample_data.parquet")

    batch_size = 1_500_000
    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.mode = PipelineModes.OTHER
    config.edge_buffer_size = 2
    config.pipeline_batch_size = batch_size
    config.num_threads = os.cpu_count()

    configure_logging(log_level=logging.INFO)

    pipeline = LinearPipeline(config)

    pipeline.set_source(
        ConnectionSourceStage(
            config,
            test_data,
            repeat_count=32,
            events_per_loop=batch_size
        )
    )

    pipeline.add_stage(MonitorStage(config, description="Source rate"))

    pipeline.add_stage(
        GraphConstructionStage(
            config,
            'artifacts/sample_edge_scaler.pkl'
        )
    )

    pipeline.add_stage(MonitorStage(config, description="Graph generation rate"))

    pipeline.add_stage(
        GraphInferenceStage(
            config,
            'artifacts/sample_weights.pth'
        )
    )

    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    pipeline.add_stage(
        CombinePredictionsStage(config)
    )

    pipeline.add_stage(MonitorStage(config, description="Combine rate"))

    out = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()