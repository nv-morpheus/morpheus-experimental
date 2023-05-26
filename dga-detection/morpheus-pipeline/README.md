<!--
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
-->

# Example Morpheus Pipeline for DGA Detection

Example Morpheus pipeline using Triton Inference server and Morpheus.

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```bash
docker pull nvcr.io/nvidia/tritonserver:23.03-py3
```

##### Start Triton Inference Server Container
From the `morpheus-experimental` repo root directory, run the following to launch Triton and load the `dga-detection-onnx` model:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/dga-detection/models:/models nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model dga-detection-onnx
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| dga-detection-onnx | 1       | READY  |
+------------------+---------+--------+
```

> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

### Build and Run Morpheus Container

Now that the model has been deployed successfully. For the experimental pipeline to execute, let's build a Morpheus container if one does not already exist.

**Note**: Before running the Morpheus container, we would need to supply an additional docker parameter to bind the Morpheus experimental pipeline repo to the container as a volume as shown in the example.

Build the release container as instructed in the [Build Morpheus Container] section of [Getting Started with Morpheus] document.

Set the following environmental variable from the root of your `morpheus-experimental` repo:
```bash
export MORPHEUS_EXPERIMENTAL_ROOT=$(pwd)
```

Now `cd` to your Morpheus repo and run the following to start your Morpheus container:
```bash
DOCKER_EXTRA_ARGS="-v ${MORPHEUS_EXPERIMENTAL_ROOT}:/workspace/morpheus-experimental" ./docker/run_container_release.sh
```

### Run DGA Detection Pipeline

Run the following in your container to start the DGA detection pipeline:

```bash
cd morpheus-experimental/dga-detection/morpheus-pipeline
python run.py \
    --log_level INFO \
    --num_threads 1 \
    --input_file ../datasets/dga-validation-data.csv \
    --output_file ./dga-detection-output.jsonlines \
    --model_name dga-detection-onnx \
    --server_url localhost:8001
```

Use `--help` to display information about the command line options:

```bash
python run.py --help

Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use.
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers.  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model.  [x>=1]
  --input_file PATH               Input filepath.  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --model_name TEXT               The name of the model that is deployed on
                                  Tritonserver.  [required]
  --model_seq_length INTEGER RANGE
                                  Sequence length to use for the model.
                                  [x>=1]
  --server_url TEXT               Tritonserver url.  [required]
  --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                  Specify the logging level to use.
  --help                          Show this message and exit.
```

