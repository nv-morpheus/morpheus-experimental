# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Args used in FROM commands must come first
ARG FROM_IMAGE="gpuci/miniforge-cuda"
ARG CUDA_VER=11.8
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:${CUDA_VER}-devel-${LINUX_DISTRO}${LINUX_VER} AS base

# Required arguments
ARG IMAGE_TYPE=base
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=23.02
ARG PYTHON_VER=3.10
ARG CONDA_CHANNEL=rapidsai

# Temp option to set conda channel_alias property during build
ARG CONDA_CHANNEL_ALIAS

# Capture argument used for FROM
ARG CUDA_VER

# Install dependencies to build
RUN apt-get update &&\
    apt-get upgrade -y &&\
    curl -sL https://deb.nodesource.com/setup_16.x | bash - &&\
    apt-get install --no-install-recommends -y \
        build-essential pkg-config curl unzip tar zip openssh-client bc jq nodejs git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# All code will be under /workspace
WORKDIR /workspace

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    conda config --env --add channels conda-forge &&\
    # Install mamba, boa and git here. Conda build breaks with other git installs
    /opt/conda/bin/conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" "git >=2.35.3" "git-lfs" "python=${PYTHON_VER}" &&\
    source activate base &&\
    git lfs install
    # conda clean -afy

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM base as runtime

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n mor_exp -c rapidsai -c conda-forge -c nvidia \
    rapids=${RAPIDS_VER} python=${PYTHON_VER} cudatoolkit=${CUDA_VER} &&\
    # Clean and activate
    # conda clean -afy && \
    sed -i 's/conda activate base/conda activate mor_exp/g' ~/.bashrc

# Set the permenant conda channels to use for morpheus experimental
RUN source activate mor_exp &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/workspace/docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

COPY docker/conda/environments/requirements.txt ./docker/conda/environments/
COPY docker/conda/environments/cuda${CUDA_VER}_runtime.yml ./docker/conda/environments/

RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked
RUN source activate mor_exp
RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env update -n mor_exp --file ./docker/conda/environments/cuda${CUDA_VER}_runtime.yml
RUN conda clean -afy

# Only copy specific files/folders over that are necessary for runtime
COPY "./docker" "./docker"
COPY "./anomalous-auth-detection" "./anomalous-auth-detection"
COPY "./appshield-dga-detection" "./appshield-dga-detection"
COPY "./asset-clustering" "./asset-clustering"
COPY "./dga-detection" "./dga-detection"
COPY "./ids-detection" "./ids-detection"
COPY "./log-sequence-ad" "./log-sequence-ad"
COPY "./operational-technology" "./operational-technology"
COPY "./phishing-url-detection" "./phishing-url-detection"
COPY "./string-resemblance-grouping" "./string-resemblance-grouping"
COPY ["*.md", "LICENSE", "./"]

# Add SRG to path to make importing and usage easier
ENV PYTHONPATH="/workspace/string-resemblance-grouping:$PYTHONPATH"

# remedy for CVE-2015-20107
RUN find / -name '*mailcap*.*py*' | xargs rm