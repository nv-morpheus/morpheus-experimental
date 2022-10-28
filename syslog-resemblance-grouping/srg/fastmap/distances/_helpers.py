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


def _to_dict(x):
    if isinstance(x, dict):
        return x
    else:
        return {idx: count for (idx, count) in enumerate(x) if count != 0}


def _match_vec_inputs(x, y):

    if type(x) == type(y):
        return x, y
    elif isinstance(x, dict) or isinstance(y, dict):
        d, coll = (x, y) if isinstance(x, dict) else (y, x)
        return d, _to_dict(coll)
    else:
        assert len(x) == len(y), "Non-set, non-dict list-like inputs must have the same length"
        return x, y
