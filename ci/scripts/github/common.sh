#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function print_env_vars() {
    rapids-logger "Environ:"
    env | grep -v -E "AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|GH_TOKEN" | sort
}

rapids-logger "Env Setup"
print_env_vars
rapids-logger "---------"
mkdir -p ${WORKSPACE_TMP}
source /opt/conda/etc/profile.d/conda.sh
export MORPHEUS_EXPERIMENTAL_ROOT=${MORPHEUS_EXPERIMENTAL_ROOT:-$(git rev-parse --show-toplevel)}
cd ${MORPHEUS_EXPERIMENTAL_ROOT}

# For non-gpu hosts nproc will correctly report the number of cores we are able to use
# On a GPU host however nproc will report the total number of cores and PARALLEL_LEVEL
# will be defined specifying the subset we are allowed to use.
NUM_CORES=$(nproc)
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-${NUM_CORES}}
rapids-logger "Procs: ${NUM_CORES}"
/usr/bin/lscpu

rapids-logger "Memory"
/usr/bin/free -g

rapids-logger "User Info"
id

# For PRs, $GIT_BRANCH is like: pull-request/989
REPO_NAME=$(basename "${GITHUB_REPOSITORY}")
ORG_NAME="${GITHUB_REPOSITORY_OWNER}"
PR_NUM="${GITHUB_REF_NAME##*/}"


function create_conda_env() {
    rapids-logger "Creating conda env"
    conda config --add pkgs_dirs /opt/conda/pkgs
    conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
    mamba env create -q -n morpheus-experimental -f ${MORPHEUS_EXPERIMENTAL_ROOT}/ci/conda_env/ci.yml

    conda activate morpheus-experimental

    rapids-logger "Final Conda Environment"
    show_conda_info
}

function fetch_base_branch() {
    rapids-logger "Retrieving base branch from GitHub API: ${GITHUB_API_URL}/repos/${ORG_NAME}/${REPO_NAME}/pulls/${PR_NUM}"
    [[ -n "$GH_TOKEN" ]] && CURL_HEADERS=('-H' "Authorization: token ${GH_TOKEN}")
    RESP=$(
    curl -s \
        -H "Accept: application/vnd.github.v3+json" \
        "${CURL_HEADERS[@]}" \
        "${GITHUB_API_URL}/repos/${ORG_NAME}/${REPO_NAME}/pulls/${PR_NUM}"
    )

    BASE_BRANCH=$(echo "${RESP}" | jq -r '.base.ref')

    # Change target is the branch name we are merging into but due to the weird way jenkins does
    # the checkout it isn't recognized by git without the origin/ prefix
    export CHANGE_TARGET="origin/${BASE_BRANCH}"
    rapids-logger "Base branch: ${BASE_BRANCH}"
}

function show_conda_info() {

    rapids-logger "Check Conda info"
    conda info
    conda config --show-sources
    conda list --show-channel-urls
}
