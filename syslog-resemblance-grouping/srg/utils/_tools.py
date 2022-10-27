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

from collections import abc
import numpy as np
from typing import Set
from math import ceil


def gen_flatten(iterables):
    """
    General purpose collection flattener that returns a list of the objects from the second-level iterables.
    Parameters
    ----------
    iterables: The iterable of iterables to flatten

    Returns
    -------
    List
    """
    flattened = (elem for iterable in iterables for elem in iterable)
    return list(flattened)


def shingler(s, shingle_size: int) -> Set[str]:
    """
    Shingles the string of the input by the parameterized shingle-size.
    Parameters
    ----------
    s: The string (or string-like) object to shingle
    shingle_size: The size of the character shingles

    Returns
    -------
    Set[str]: The set of shingled character strings
    """
    input_string = str(s)
    if shingle_size >= len(input_string):
        return set(input_string)
    return set([input_string[i:i+shingle_size] for i in range(len(input_string) - shingle_size + 1)])


def is_list_like(obj) -> bool:
    """
    Adapted from Pandas is_list_like. Excludes dict and sets to focus on ordered list-like objects only.

    Parameters
    ----------
    obj: The object to test for list-likeness

    Returns
    -------
    bool
    """
    return (
        isinstance(obj, abc.Iterable)
        and not isinstance(obj, (str, bytes, dict, set))
        and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
    )
    

def create_x_axis(start, stop, step=0.01):
    return [start + i * step for i in range(int(ceil((stop - start)/step)))]


def find_local_min(x, y):
    minima = []
    for idx in range(1, len(x) - 1):
        if y[idx] <= y[idx-1] and y[idx] < y[idx+1]:
            minima.append(x[idx])
    return minima


def find_local_max(x, y):
    return find_local_min(x, [-y_i for y_i in y])
    