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

import argparse
import os
from datetime import datetime

from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score

from .platemail import process_email


# Get the path to the directory this file is in
os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser(description="Process and run phishing inference over emails loaded into a csv with the body, sender, and arrival time")
parser.add_argument('--emails_path', default=os.path.abspath('../datasets/test/20230510_example_test.csv'), required=False, help="The path to the csv containing emails")
parser.add_argument('--phishing_model', default=os.path.abspath('../models/phishing_model/20230510_phishing.onnx'), required=False, help="The path to the phishing model to inference")
parser.add_argument('--body_col', default='body', required=False, help="The column containing the email body data")
parser.add_argument('--time_col', default='arrival_time', required=False, help="The column containing the arrival time data")
parser.add_argument('--sender_col', default='sender', required=False, help="The column containing the sender data")

_INTENT_MODELS = [{'intent': 'money', 'path': os.path.abspath('../models/intent_models/moneyv2_checkpoint-2167')},
                  {'intent': 'banking', 'path': os.path.abspath('../models/intent_models/personalv2_checkpoint-2167')}
                  ]


def _vectorize(parsed_data):
    X = []
    for entry in parsed_data:
        vector = []
        for _, results in entry['intents'].items():
            vector.append(results['id'])
        vector.append(entry['new_intent'])
        vector.append(entry['syntax_sim'])
        vector.append(entry['time_liklihood'])
        X.append(vector)
    return np.array(X).astype(np.float32)


class ProcessPlatemail:

    def __init__(self, emails_path, phishing_model, body_col='body', time_col='arrival_time', sender_col='sender'):
        self._path = emails_path
        self._phishing_model = phishing_model
        self._body_col = body_col
        self._time_col = time_col
        self._sender_col = sender_col

    def run(self):
        print("Running Platemail...")
        emails = pd.read_csv(self._path)
        emails.rename(columns={self._body_col: 'body', self._time_col: 'arrival_time', self._sender_col: 'sender'}, inplace=True)
        parsed_emails = emails.to_dict('records')
        for model_dict in _INTENT_MODELS:
            inferences = []
            classifier = pipeline('text-classification', model=model_dict['path'], truncation=True, max_length=512)
            label_map = classifier.model.config.label2id
            for out in tqdm(classifier([parsed_email['body'] for parsed_email in parsed_emails], batch_size=256), total=len(parsed_emails)):
                id = label_map[out['label']]
                inferences.append({'intent': model_dict['intent'], 'label': out['label'], 'id': id, 'score': out['score']})
            for p, i in zip(parsed_emails, inferences):
                intent_dict = {i['intent']: {'label': i['label'], 'id': i['id'], 'score':i['score']}}
                if 'intents' in p:
                    if isinstance(p['intents'], dict):
                        p['intents'].update(intent_dict)
                    else:
                        p['intents'] = intent_dict
                else:
                    p['intents'] = intent_dict

        for entry in tqdm(parsed_emails, total=len(parsed_emails)):
            entry['arrival_time'] = datetime.fromisoformat(entry['arrival_time'])
            entry.update(process_email(entry))

        tensor = _vectorize(parsed_emails)

        sess = ort.InferenceSession(self._phishing_model, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        preds = sess.run([label_name], {input_name: tensor})[0]

        for entry, pred, score in tqdm(zip(parsed_emails, list(preds), list(scores)), total=len(parsed_emails)):
            entry['prediction'] = pred
            entry['score'] = score

        return parsed_emails


def main():
    opt = parser.parse_args()
    platemail = ProcessPlatemail(emails_path=opt.emails_path, phishing_model=opt.phishing_model, body_col=opt.body_col, time_col=opt.time_col, sender_col=opt.sender_col)
    equip_platemail = platemail.run()
    df = pd.DataFrame(equip_platemail)
    print(df[['label', 'prediction']])
    acc = accuracy_score(df['label'], df['prediction'])
    print("Platemail successful. Model accuracy: %f" % acc)


if __name__ == '__main__':
    main()