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

import abc


class InputError(Exception):
    """
    Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message


class Distance(metaclass=abc.ABCMeta):
    """
    Base abstract class for distance metrics to be utilized by the FastMap method.
    """
    @staticmethod
    @abc.abstractmethod
    def get_name(self):
        """
        Return the name of the distance metric.
        """
        pass

    @abc.abstractmethod
    def calculate(self, x, y) -> float:
        """
        Calculate the distance between two objects.

        Parameters
        ----------
        x: object
        y: object

        Returns
        -------
        float
        """
        pass
