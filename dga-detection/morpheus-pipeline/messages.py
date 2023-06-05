# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import typing

from morpheus.messages import MultiInferenceMessage
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta


@dataclasses.dataclass
class MultiInferenceDGAMessage(MultiInferenceMessage, cpp_class=None):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["domains", "seq_ids"]

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count)

    @property
    def domains(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self._get_tensor_prop("domains")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of which inference requests belong to each message.

        Returns
        -------
        cupy.ndarray
            Ids used to index from an inference input to a message. Necessary since there can be more
            inference inputs than messages (i.e., if some messages get broken into multiple inference requests).

        """

        return self._get_tensor_prop("seq_ids")
