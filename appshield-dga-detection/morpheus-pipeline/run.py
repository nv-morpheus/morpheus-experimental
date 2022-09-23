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

import logging

import click
import psutil
from morpheus.config import Config, CppConfig, PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import \
    TritonInferenceStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.logger import configure_logging

from create_feature import CreateFeatureDGAStage
from preprocessing import PreprocessingDGAStage


@click.command()
@click.option("--use_cpp", default=False, help="Default value is False")
@click.option(
    "--num_threads",
    default=psutil.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=100000,
    type=click.IntRange(min=1),
    help=(
        "Internal batch size for the pipeline. Can be much larger than the model batch size. "
        "Also used for Kafka consumers"
    ),
)
@click.option(
    "--model_max_batch_size",
    default=256,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=75,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--model_name",
    default="dga-appshield-cnn-onnx",
    help="The name of the model that is deployed on Tritonserver",
)
@click.option("--server_url", required=True, help="Tritonserver url")
@click.option(
    "--input_glob",
    type=click.STRING,
    required=True,
    help="Input files",
)
@click.option(
    "--tokenizer_path",
    type=click.STRING,
    required=True,
    help="Tokenizer path",
)
@click.option(
    "--output_file",
    type=click.STRING,
    default="ransomware_detection_output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--watch_directory",
    type=bool,
    default=False,
    help=(
        "The watch directory option instructs this stage to not close down once all files have been read. "
        "Instead it will read all files that match the 'input_glob' pattern, and then continue to watch "
        "the directory for additional files. Any new files that are added that match the glob will then "
        "be processed."
    ),
)
def run_pipeline(
    use_cpp,
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    model_name,
    server_url,
    input_glob,
    tokenizer_path,
    output_file,
    watch_directory,
):

    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for Pipeline mode
    CppConfig.set_should_use_cpp(use_cpp)
    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.class_labels = ["probs"]
    config.edge_buffer_size = 4

    kwargs = {}

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    cols_interested_plugins = [
        "PID",
        "Process",
        "Heap",
        "Virtual-address",
        "URL",
        "plugin",
        "snapshot_id",
        "timestamp",
        "Domain",
    ]

    DOMAIN_LEN = 75
    feature_columns = ["char_" + str(i) for i in range(DOMAIN_LEN)]
    interested_plugins = ["urls"]

    # input_glob = os.path.join(input_dir, "snapshot-*", "*.json")

    # Set source stage
    pipeline.set_source(
        AppShieldSourceStage(
            config,
            input_glob,
            interested_plugins,
            cols_interested_plugins,
            watch_directory=watch_directory,
        )
    )
    # Add a monitor stage
    # pipeline.add_stage(MonitorStage(config, description="from-file rate", unit="inf"))

    # Add processing stage
    pipeline.add_stage(
        CreateFeatureDGAStage(
            config,
            feature_columns=feature_columns,
            required_plugins=interested_plugins,
            tokenizer_path=tokenizer_path,
        )
    )

    # # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Create features rate"))

    pipeline.add_stage(PreprocessingDGAStage(config, feature_columns=feature_columns))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
        )
    )
    # # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # Add a add classification stage
    pipeline.add_stage(AddScoresStage(config, labels=["probs"]))

    # Add a monitor stage
    pipeline.add_stage(
        MonitorStage(config, description="Add classification rate", unit="add-class")
    )

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(SerializeStage(config, **kwargs))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Serialize rate", unit="ser"))

    # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage
    pipeline.add_stage(
        MonitorStage(config, description="Write to file rate", unit="to-file")
    )

    # Build pipeline
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
