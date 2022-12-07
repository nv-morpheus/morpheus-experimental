#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/mor_exp"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0)-runtime"}
DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}

# Build args
FROM_IMAGE=${FROM_IMAGE:-"gpuci/miniforge-cuda"}
CUDA_VER=${CUDA_VER:-11.5}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-20.04}
RAPIDS_VER=${RAPIDS_VER:-22.08}
PYTHON_VER=${PYTHON_VER:-3.8}

DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --target ${DOCKER_TARGET}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE=${FROM_IMAGE}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_VER=${CUDA_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_DISTRO=${LINUX_DISTRO}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_VER=${LINUX_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg RAPIDS_VER=${RAPIDS_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg PYTHON_VER=${PYTHON_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT=1

echo "Building morpheus experimental:${DOCKER_TAG}..."
echo "   FROM_IMAGE      : ${FROM_IMAGE}"
echo "   CUDA_VER        : ${CUDA_VER}"
echo "   LINUX_DISTRO    : ${LINUX_DISTRO}"
echo "   LINUX_VER       : ${LINUX_VER}"
echo "   RAPIDS_VER      : ${RAPIDS_VER}"
echo "   PYTHON_VER      : ${PYTHON_VER}"
echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f docker/Dockerfile ."
echo "   Note: add '--progress plain' to DOCKER_ARGS to show all container build output"

docker build ${DOCKER_ARGS} -f docker/Dockerfile .