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
ARG FROM_IMAGE="nvidia/cuda"
ARG CUDA_MAJOR_VER=11
ARG CUDA_MINOR_VER=8
ARG CUDA_REV_VER=0
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=22.04

# =====Global Options=====
ARG PYTHON_VER=3.10
# Allows running the docker build from a different directory than MORPHEUS_ROOT. Needed for using Morpheus in a
# submodule

# Supply a channel alias to use for conda. This is needed if the conda channels go down
ARG CONDA_CHANNEL_ALIAS="https://conda.anaconda.org"

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}.${CUDA_REV_VER}-base-${LINUX_DISTRO}${LINUX_VER} AS base

ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG PYTHON_VER

ENV DEBIAN_FRONTEND=noninteractive

# Remove the file which breaks apt caching since we will handle it ourselves. See
# https://vsupalov.com/buildkit-cache-mount-dockerfile/
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install dependencies to build
RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install --no-install-recommends -y \
        bc \
        build-essential \
        curl \
        git-lfs \
        jq \
        nodejs \
        openssh-client \
        pkg-config \
        tar \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

ENV CUDA_PATH=/usr/local/cuda

# All code will be under /workspace
ENV MORPHEUS_ROOT=/workspace
WORKDIR ${MORPHEUS_ROOT}

COPY --from=condaforge/mambaforge:latest /opt/conda /opt/conda

ENV PATH="${PATH}:/opt/conda/bin"

# Install mamba to speed the solve up
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    # Install mamba, boa and git here. Conda build breaks with other git installs
    /opt/conda/bin/mamba install -y -n base -c conda-forge "boa" "git >=2.35.3" "python=${PYTHON_VER}" "tini=0.19" &&\
    source activate base
    # conda clean -afy

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM base as runtime

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n mor_exp -c conda-forge  \
      python=${PYTHON_VER} && \
    # Clean and activate
    # conda clean -afy && \
    conda init bash && \
    echo "conda activate mor_exp" >> ~/.bashrc

# Set the permenant conda channels to use for morpheus-experimental
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
COPY docker/conda/environments/cuda${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}_runtime.yml ./docker/conda/environments/

RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked
RUN source activate mor_exp
RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env update -n mor_exp --file ./docker/conda/environments/cuda${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}_runtime.yml
RUN conda clean -afy

# Only copy specific files/folders over that are necessary for runtime
COPY "./docker" "./docker"
COPY "./anomalous-auth-detection" "./anomalous-auth-detection"
COPY "./appshield-dga-detection" "./appshield-dga-detection"
COPY "./asset-clustering" "./asset-clustering"
COPY "./dga-detection" "./dga-detection"
COPY "./gnn-asset-clustering" "./gnn-asset-clustering"
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