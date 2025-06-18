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

import os
import hashlib
import pandas as pd
import re
import mmh3
from math import sqrt, pow
import json

from .intent_inferencing import _INTENT_MODELS

os.chdir(os.path.dirname(os.path.realpath(__file__)))
SKETCHES = os.path.abspath('../sender_sketches')
USER_HASH_KEY = 23
_HRS_PER_WEEK = 24 * 7

# def _infer_intents(body):
#     #Pass the body of the text through the pre-defined intent models, returning a dictionary of {intent type: inference result}
#     pass

def _compare_syntax(body, syntax_sketch):
    sketch = dict(pd.read_csv(syntax_sketch).to_dict('tight')['data'])
    incoming = dict()
    stripped_body = re.sub(r'([^a-zA-Z\s]+?)', '', body)
    for token in stripped_body.split():
        if len(token) > 3:
            token_hash = mmh3.hash(token.strip().lower(), USER_HASH_KEY, signed=False)
            incoming[token_hash] = incoming.get(token_hash, 0) + 1
    dot = sum([a_i * sketch.get(i, 0) for (i, a_i) in incoming.items()])
    s_norm = sqrt(sum([pow(b_i, 2.0) for b_i in sketch.values()]))
    inc_norm = sqrt(sum([pow(a_i, 2.0) for a_i in incoming.values()]))
    divisor = (s_norm * inc_norm)
    if divisor == 0:
        cos_sim = 0
    else:
        cos_sim = dot / (s_norm * inc_norm)
    return cos_sim

#I've binned the timestamps into the nearest hour and then taken that mod the number of hours in a week. This maps all the stamps to the corresponding hour of the week that the email was sent.
#For the actual feature, we're just looking at the empirical probability that the incoming email was sent during that particular hour. In the future, we may be able to do something with a little more granularity using
# kernel density estimators, but this was a simpler and faster statistical approach to what this feature was trying to capture.
def _compare_time(date, time_sketch):
    times = pd.read_csv(time_sketch)['time'].values.tolist()
    if len(times) < 20:
        return -1
    else:
        binned_time = int(round(date.timestamp()) // (60 * 60) % _HRS_PER_WEEK)
        timestamps = [int(round(ts) // (60 * 60) % _HRS_PER_WEEK) for ts in times]
        observed_bins = [t for t in timestamps if t == binned_time]
        return len(observed_bins) / len(timestamps)


def process_email(parsed_email):
    body = parsed_email['body']
    sender = parsed_email['sender']
    date = parsed_email['arrival_time']

    base_file = hashlib.sha256(sender.encode('utf8')).hexdigest()
    sender_meta = os.path.join(SKETCHES, base_file+'_meta.json')
    syntax_sketch = os.path.join(SKETCHES, base_file+'_syntax.csv')
    time_sketch = os.path.join(SKETCHES, base_file+'_time.csv')

    syntax_sketch_exists = os.path.isfile(syntax_sketch)
    time_sketch_exists = os.path.isfile(time_sketch)
    sender_meta_exists = os.path.isfile(sender_meta)

    if sender_meta_exists:
        with open(sender_meta, 'r') as meta_in:
            meta = json.load(meta_in)
        new_intent = 0
        for models in _INTENT_MODELS:
            intent_name = models['intent']
            incoming_intent = parsed_email['intents'].get(intent_name, {'id': 1})['id']
            sketch_intent = meta.get(intent_name, 1)
            if incoming_intent < sketch_intent:
                new_intent = 1
        parsed_email['new_intent'] = new_intent
    else:
        parsed_email['new_intent'] = -1
    if syntax_sketch_exists:
        syntax_sim = _compare_syntax(body, syntax_sketch)
        with open(sender_meta, 'r') as meta_in:
            meta = json.load(meta_in)
        mu = meta['syntax_avg']
        std = meta['syntax_std']
        # We don't care if syntax sim is above mu, it just means that the email is very similar to the historical sketch of the sender. We also need a value for when we don't have historical info. By flipping the usual
        # normalization, we can use -1 as our catch all, since negative values will be above average to a point where the final classifier will have nothing really to use when it's in the negative range, i.e., above avg or no data
        parsed_email['syntax_sim'] = (mu - syntax_sim) / std
    else:
        parsed_email['syntax_sim'] = -1

    if time_sketch_exists:
        time_liklihood = _compare_time(date, time_sketch)
        parsed_email['time_liklihood'] = time_liklihood
    else:
        parsed_email['time_liklihood'] = -1

    return parsed_email
