#!/bin/bash
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


# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/mor_exp"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0)-runtime"}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

DOCKER_ARGS="--env WORKSPACE_VOLUME=${PWD} --net=host --gpus=all --cap-add=sys_nice ${DOCKER_EXTRA_ARGS}"
echo -e "${g}Args: ${DOCKER_ARGS}${x}"

echo -e "${g}Launching ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${x}"

docker run --rm -ti ${DOCKER_ARGS} ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} "${@:-bash}"