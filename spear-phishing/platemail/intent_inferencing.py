# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformers import pipeline
import os


# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(__file__))
_INTENT_MODELS = [{'intent': 'money', 'path': os.path.join(BASEDIR, 'intent_models/moneyv2_checkpoint-2167')},
                  {'intent': 'banking', 'path': os.path.join(BASEDIR, 'intent_models/personalv2_checkpoint-2167')},
                  {'intent': 'crypto', 'path': os.path.join(BASEDIR, 'intent_models/crypto_checkpoint-2362')}
                  ]

#Returns {intent_class: {'label': label, 'id': int id, 'score': float score}}
def infer_intents(body):
    results = dict()
    for model_dict in _INTENT_MODELS:
        classifier = pipeline('text-classification', model=model_dict['path'], truncation=True, max_length=512)
        inference = classifier(body)[0]
        id = classifier.model.config.label2id[inference['label']]
        results[model_dict['intent']] = {'label': inference['label'], 'id': id, 'score': inference['score']}
    return results
