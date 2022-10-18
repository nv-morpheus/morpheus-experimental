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

# set -x

export SCRIPT_DIR=${SCRIPT_DIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"}
export REPO_DIR=$(realpath ${REPO_DIR:-"${SCRIPT_DIR}/../.."})
export PY_ROOT="${REPO_DIR}"
export PY_CFG="${PY_ROOT}/setup.cfg"
export PY_DIRS="${PY_ROOT} ci/scripts"

# work-around for known yapf issue https://github.com/google/yapf/issues/984
export YAPF_EXCLUDE_FLAGS=""

# Determine the commits to compare against. If running in CI, these will be set. Otherwise, diff with main
export BASE_SHA=${CHANGE_TARGET:-${BASE_SHA:-$(${SCRIPT_DIR}/gitutils.py get_merge_target)}}
export COMMIT_SHA=${GIT_COMMIT:-${COMMIT_SHA:-HEAD}}

export PYTHON_FILE_REGEX='^(\.\/)?(?!\.|build).*\.(py|pyx|pxd)$'

# Use these options to skip any of the checks
export SKIP_COPYRIGHT=${SKIP_COPYRIGHT:-""}
export SKIP_ISORT=${SKIP_ISORT:-""}
export SKIP_YAPF=${SKIP_YAPF:-""}

# Set BUILD_DIR to use a different build folder
export BUILD_DIR=${BUILD_DIR:-"${REPO_DIR}/build"}

# Speficy the clang-tools version to use. Default 14
export CLANG_TOOLS_VERSION=${CLANG_TOOLS_VERSION:-14}

# Determine the merge base as the root to compare against. Optionally pass in a
# result variable otherwise the output is printed to stdout
function get_merge_base() {
   local __resultvar=$1
   local result=$(git merge-base ${BASE_SHA:-main} ${COMMIT_SHA:-HEAD})

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'${result}'"
   else
      echo "${result}"
   fi
}

# Determine the changed files. First argument is the (optional) regex filter on
# the results. Second argument is the (optional) variable with the returned
# results. Otherwise the output is printed to stdout. Result is an array
function get_modified_files() {
   local  __resultvar=$2

   local GIT_DIFF_ARGS=${GIT_DIFF_ARGS:-"--name-only"}
   local GIT_DIFF_BASE=${GIT_DIFF_BASE:-$(get_merge_base)}

   # If invoked by a git-commit-hook, this will be populated
   local result=( $(git diff ${GIT_DIFF_ARGS} $(get_merge_base) | grep -P ${1:-'.*'}) )

   local files=()

   for i in "${result[@]}"; do
      if [[ -e "${i}" ]]; then
         files+=(${i})
      fi
   done

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="( ${files[@]} )"
   else
      echo "${files[@]}"
   fi
}

# Determine a unified diff useful for clang-XXX-diff commands. First arg is
# optional file regex. Second argument is the (optional) variable with the
# returned results. Otherwise the output is printed to stdout
function get_unified_diff() {
   local  __resultvar=$2

   local result=$(git diff --no-color --relative -U0 $(get_merge_base) -- $(get_modified_files $1))

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'${result}'"
   else
      echo "${result}"
   fi
}


function get_num_proc() {
   NPROC_TOOL=`which nproc`
   NUM_PROC=${NUM_PROC:-`${NPROC_TOOL}`}
   echo "${NUM_PROC}"
}

function cleanup {
   # Restore the original directory
   popd &> /dev/null
}

trap cleanup EXIT

# Change directory to the repo root
pushd "${REPO_DIR}" &> /dev/null
