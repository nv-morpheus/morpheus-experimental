<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DGA Detection Morpheus Pipeline for AppShield Data

Experimental Morpheus pipeline using Docker containers for Triton Inference server and Morpheus SDK/Client.

## Setup Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from [NGC] suitable for your environment.

```bash
docker pull nvcr.io/nvidia/tritonserver:22.08-py3
```

##### Start Triton Inference Server container

Change to a pipeline directory:

```bash
cd ${MORPHEUS_EXPERIMENTAL_ROOT}/appshield-dga-detection/morpheus-pipeline
```

# Run Triton in explicit mode
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models/triton-model-repo nvcr.io/nvidia/tritonserver:22.08-py3 \
   tritonserver --model-repository=/models/triton-model-repo \
                --exit-on-error=false \
                --model-control-mode=explicit \
                --load-model dga-appshield-cnn
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
I0922 16:01:56.339664 1 server.cc:626] 
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| dga-appshield-cnn | 1       | READY  |
+-------------------+---------+--------+
```

##### Build and Run Morpheus Container

Now that the model has been deployed successfully. For the experimental pipeline to execute, let's build a Morpheus container if one does not already exist.

**Note**: Before running the Morpheus container, we would need to supply an additional docker parameter to bind the Morpheus experimental pipeline repo to the container as a volume.

Example:
```bash
DOCKER_EXTRA_ARGS="-v ${MORPHEUS_EXPERIMENTAL_ROOT}:/workspace/morpheus_experimental" ./docker/run_container_release.sh
```

Build the dev container as instructed in the [Build Morpheus Container] section of [Getting Started with Morpheus] document.

## Requirements
**Note**: Make sure below dependencies are installed in your environment before running the DGA detection pipeline. Run the installation command specified below if not.

```bash
pip install dask==2022.9.0 distributed==2022.9.0 tensorflow==2.10.0 tldextract==3.3.1
```

Install 

## Run Pipeline
Launch the example using the following

```bash
cd ${MORPHEUS_ROOT}/morpheus_experimental/appshield-dga-detection/morpheus-pipeline

python run.py --server_url=localhost:8001 \
              --model_name=dga-appshield-cnn \
              --input_glob=${MORPHEUS_ROOT}/morpheus_experimental/data/URLS_Snapshots/snapshot-*/*.json \
              --tokenizer_path=./tokenizer.csv \
              --output_file=./dga_detection_output.jsonlines
```

The configuration options for this example can be queried with:

```bash
python run.py --help
```

```
Usage: run.py [OPTIONS]

Options:
  --use_cpp BOOLEAN
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model  [x>=1]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model  [x>=1]
  --model_name TEXT               The name of the model that is deployed on
                                  Tritonserver
  --server_url TEXT               Tritonserver url  [required]
  --input_glob TEXT               Input files  [required]
  --tokenizer_path TEXT           Tokenizer path  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --help                          Show this message and exit.
  ```

  [NGC]: https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver
  [Getting Started with Morpheus]:https://github.com/nv-morpheus/Morpheus#getting-started-with-morpheus
  [Build Morpheus Container]: https://github.com/nv-morpheus/Morpheus#build-morpheus-container