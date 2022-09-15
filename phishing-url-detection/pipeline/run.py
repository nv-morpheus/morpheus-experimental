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


@click.command()
@click.option(
    "--num_threads",
    default=psutil.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=10000,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=500,  #297,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--model_name",
    default="phishurl-appshield-combined-lstm-dnn-onnx",
    help="The name of the model that is deployed on Tritonserver",
)
@click.option("--server_url", required=True, help="Tritonserver url")
@click.option(
    "--input_glob",
    type=click.STRING,
    required=True,
    help="Input glob",
)

@click.option(
    "--tokenizer_path",
    type=click.STRING,
    required=True,
    help="Tokenizer path",
)

@click.option(
    "--max_min_norm_path",
    type=click.STRING,
    required=True,
    help="Max min normalization path",
)
@click.option(
    "--output_file",
    type=click.STRING,
    default="ransomware_detection_output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 model_fea_length,
                 model_name,
                 server_url,
                 input_glob,
                 tokenizer_path,
                 max_min_norm_path,
                 output_file):

    from morpheus.config import Config
    from morpheus.config import PipelineModes
    from morpheus.utils.logging import configure_logging

    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for FIL mode
    config = Config.get()
    config.use_cpp = False
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    # Setting number of threads to 1 adding
    num_threads = 1
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.class_labels = ["probs"]

    from create_feature import CreateFeatureURLStage
    from from_appshield import AppShieldSourceStage
    from inference_triton import TritonInferenceStage
    from preprocessing import PreprocessingURLStage

    from morpheus.pipeline.general_stages import AddScoresStage
    from morpheus.pipeline.general_stages import AddClassificationsStage
    from morpheus.pipeline.general_stages import MonitorStage
    from morpheus.pipeline.output.serialize import SerializeStage
    from morpheus.pipeline.output.to_file import WriteToFileStage

    kwargs = {}

    from morpheus.pipeline.pipeline import LinearPipeline

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    raw_feature_columns = ['PID', 'Process', 'Heap', 'Virtual-address', 'URL', 'plugin', 'snapshot_id', 'timestamp']

    feature_columns = [
        'domain_in_alexa',
        'domain_len',
        'domain_numbers',
        'domain_isalnum',
        'subdomain_len',
        'subdomain_numbers_count',
        'subdomain_parts_count',
        'tld_len',
        'tld_parts_count',
        'queries_amount',
        'fragments_amount',
        'path_len',
        'path_slash_counts',
        'path_double_slash_counts',
        'brand_in_subdomain',
        'brand_in_path',
        'path_max_len'
    ]
    MAX_LEN = 500
    feature_columns.extend(['word_' + str(i) for i in range(MAX_LEN)])
    required_plugins = ['urls']

    # Set source stage
    pipeline.set_source(
        AppShieldSourceStage(config,
                             input_glob,
                             watch_directory=False,
                             raw_feature_columns=raw_feature_columns,
                             required_plugins=required_plugins))
    # Add a monitor stage
    # pipeline.add_stage(MonitorStage(config, description="from-file rate", unit="inf"))

    # Add processing stage
    pipeline.add_stage(CreateFeatureURLStage(config, feature_columns=feature_columns,
                                             required_plugins=required_plugins, tokenizer_path=tokenizer_path, max_min_norm_path=max_min_norm_path))
    #pipeline.add_stage(URLPreprocessingStage(config))

    # # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Create features rate"))

    pipeline.add_stage(PreprocessingURLStage(config, feature_columns=feature_columns))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(config, model_name=model_name, server_url=server_url, force_convert_inputs=True))
    
    # # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # Add a add classification stage
    #pipeline.add_stage(AddClassificationsStage(config, labels=["probs"]))

    # Add a monitor stage
    #pipeline.add_stage(MonitorStage(config, description="Add classification rate", unit="add-class"))

    # Add a scores stage
    pipeline.add_stage(AddScoresStage(config, labels=["probs"]))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Add score rate", unit="add-score"))

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(SerializeStage(config, **kwargs))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Serialize rate", unit="ser"))

    # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Write to file rate", unit="to-file"))

    # Build pipeline
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
