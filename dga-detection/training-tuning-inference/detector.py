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
from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

GPU_COUNT = torch.cuda.device_count()


class Detector(ABC):

    def __init__(self, lr=0.001):
        self.lr = lr
        self._model = None
        self._optimizer = None
        self._criterion = nn.CrossEntropyLoss()

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @abstractmethod
    def init_model(self, char_vocab, hidden_size, n_domain_type, n_layers):
        pass

    @abstractmethod
    def train_model(self, training_data, labels, batch_size=1000, epochs=1, train_size=0.7):
        pass

    @abstractmethod
    def predict(self, epoch, train_dataset):
        pass

    def load_model(self, file_path):
        """
        This function load already saved model and sets cuda parameters.

        Parameters
        ----------
        file_path : str
            File path of a model to be loaded
        """

        model = torch.load(file_path)
        model.eval()
        self._model = model
        self._set_model2cuda()
        self._set_optimizer()

    def save_model(self, file_path):
        """ This function saves model to a given location.

        :param file_path: File path of a model to be saved.
        :type file_path: string
        """

        torch.save(self.model, file_path)

    def _save_checkpoint(self, checkpoint, file_path):
        torch.save(checkpoint, file_path)
        log.info("Pretrained model checkpoint saved to location: '{}'".format(file_path))

    def _set_parallelism(self):
        if GPU_COUNT > 1:
            log.info("CUDA device count: {}".format(GPU_COUNT))
            self._model = nn.DataParallel(self.model)
            self._set_model2cuda()
        else:
            self._set_model2cuda()

    def _set_optimizer(self):
        self._optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr, weight_decay=0.0)

    def _set_model2cuda(self):
        if torch.cuda.is_available():
            log.info("Found GPU's now setting up cuda for the model")
            self.model.cuda()

    def leverage_model(self, model):
        """
        This function leverages model by setting parallelism parameters.

        Parameters
        ----------
        model : RNNClassifier
            Model instance
        """
        model.eval()
        self._model = model
        self._set_parallelism()
        self._set_optimizer()

    def get_unwrapped_model(self):
        if GPU_COUNT > 1:
            model = self.model.module
        else:
            model = self.model
        return model
