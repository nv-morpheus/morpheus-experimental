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

from srg.fastmap.distances._distance import Distance, InputError
from pandas.api.types import is_list_like
from math import sqrt, pow

from srg.fastmap.distances._helpers import _match_vec_inputs


def _d(x, y):
    if isinstance(x, dict):
        all_keys = {*x}.union({*y})
        diffs = [x.get(key, 0)-y.get(key, 0) for key in all_keys]
    else:
        diffs = [xi - yi for (xi, yi) in zip(x, y)]
    return sqrt(sum([pow(diff, 2) for diff in diffs]))


class L2(Distance):

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "L2"

    def calculate(self, x, y) -> float:

        if not (is_list_like(x, allow_sets=False) and is_list_like(y, allow_sets=False)):
            raise InputError("Cosine distance needs to be non-set, list like objects")

        x, y = _match_vec_inputs(x, y)
        d = _d(x, y)

        return d
