#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# This will run all style cleanup tools on the source code
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/../scripts/common.sh

# Get the list of modified files to check
get_modified_files ${PYTHON_FILE_REGEX} PY_MODIFIED_FILES
get_modified_files ${CPP_FILE_REGEX} CPP_MODIFIED_FILES

# Run copyright fix
if [[ "${SKIP_COPYRIGHT}" == "" ]]; then
   echo "Running copyright check..."
   python3 ./ci/scripts/copyright.py --fix-all --git-modified-only ./ 2>&1
fi


# Run isort
if [[ "${SKIP_ISORT}" == "" ]]; then
   echo "Running isort..."
   python3 -m isort --settings-file ${PY_CFG} ${PY_MODIFIED_FILES[@]} 2>&1
fi

# Run yapf
if [[ "${SKIP_YAPF}" == "" ]]; then
   echo "Running yapf..."
   python3 -m yapf -i --style ${PY_CFG} ${YAPF_EXCLUDE_FLAGS} -r ${PY_MODIFIED_FILES[@]}
fi
