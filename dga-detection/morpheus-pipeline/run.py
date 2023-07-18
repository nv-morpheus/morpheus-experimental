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

import logging
import os

import click
from preprocessing import PreprocessDGAStage

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
# from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers."),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Input filepath.",
)
@click.option(
    "--output_file",
    default="log-parsing-output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--model_name",
    required=True,
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--model_seq_length",
              default=100,
              type=click.IntRange(min=1),
              help="Sequence length to use for the model.")
@click.option("--server_url", required=True, help="Tritonserver url.")
@click.option("--log_level",
              default=logging.getLevelName(Config().log_level),
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Specify the logging level to use.")
def run_pipeline(num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 input_file,
                 output_file,
                 model_name,
                 model_seq_length,
                 server_url,
                 log_level):
    CppConfig.set_should_use_cpp(False)

    configure_logging(log_level=log_level)

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_seq_length
    config.class_labels = ["is_dga", "not_dga"]

    # Create a pipeline object.
    pipeline = LinearPipeline(config)

    # Add a source stage.
    # In this stage, messages were loaded from a file.
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False, repeat=1))

    # Add a deserialize stage.
    # At this stage, messages were logically partitioned based on the 'pipeline_batch_size'.
    pipeline.add_stage(DeserializeStage(config))

    # Add a preprocessing stage.
    # This stage preprocess the rows in the Dataframe.
    pipeline.add_stage(PreprocessDGAStage(config, column="domain"))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage.
    # This stage sends inference requests to the Tritonserver and captures the response.
    pipeline.add_stage(
        TritonInferenceStage(config, model_name=model_name, server_url=server_url, force_convert_inputs=True))

    # Add a monitor stage.
    # This stage logs the metrics (msg/sec) from the above stage.
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # pipeline.add_stage(AddClassificationsStage(config, threshold=0.5, prefix=""))

    pipeline.add_stage(AddScoresStage(config, labels=["is_dga"]))

    # Add a write file stage.
    # This stage writes all messages to a file.
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Run the pipeline.
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
