# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

log = logging.getLogger(__name__)


class DataLoader(object):
    """
    Wrapper class is used to return dataframe partitions based on batch size.

    Parameters
    ----------
    dataset : cudf.DataFrame
        Input dataframe
    batchsize : int
        Number of records in the dataframe
    """

    def __init__(self, dataset, batchsize=1000):
        self.__dataset = dataset
        self.__batchsize = batchsize

    @property
    def dataset_len(self):
        return self.__dataset.length

    @property
    def dataset(self):
        return self.__dataset

    def get_chunks(self):
        """
        A generator function that yields each chunk of original input dataframe based on batchsize

        Returns
        -------
        cudf.DataFrame
            Partitioned dataframe
        """
        prev_chunk_offset = 0
        while prev_chunk_offset < self.__dataset.length:
            curr_chunk_offset = prev_chunk_offset + self.__batchsize
            chunk = self.__dataset.data[prev_chunk_offset:curr_chunk_offset:1]
            prev_chunk_offset = curr_chunk_offset
            yield chunk
