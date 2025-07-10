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
import mmh3
import csv
import re
import pandas as pd
import json
from bacpedal.platemail._platemail import _compare_syntax

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(__file__))
SKETCHES = os.path.join(BASEDIR, 'sender_sketches')
SYNTAX_HEADER = ['token', 'count']
TIME_HEADER = ['time']
USER_HASH_KEY = 23


def _vectorize_email(body):
    stripped_body = re.sub(r'([^a-zA-Z\s]+?)', '', body)
    vector = dict()
    for token in stripped_body.split():
        if len(token) > 3:
            token_hash = mmh3.hash(token.strip().lower(), USER_HASH_KEY, signed=False)
            vector[token_hash] = vector.get(token_hash, 0) + 1
    return vector


def _dictorize_email(body):
    vector = _vectorize_email(body)
    rows = []
    for idx, val in vector.items():
        rows.append({'token': idx, 'count': val})
    return rows


def update_sender_sketch(parsed_email):
    """
    Adds or updates sketch information for senders. Looks at the sender for the provided email and
    checks for sketch data. If it exists, the sketch is updated with the new email body, intents,
    and metadata. If no sketch exists, a new one is created based on the email and saved.

    Parameters
    ----------
    parsed_email: dict[str, Any]
        The email from the sender to be added or updated to the sketch.
    """
    sender = parsed_email.get('sender', None)
    if sender:
        base_file = hashlib.sha256(sender.encode('utf8')).hexdigest()

        sender_metadata = os.path.join(SKETCHES, base_file+'_meta.json')
        syntax_sketch = os.path.join(SKETCHES, base_file+'_syntax.csv')
        time_sketch = os.path.join(SKETCHES, base_file+'_time.csv')

        if not os.path.isfile(sender_metadata):
            new_sketch = True
            meta = {'sender_hash': base_file, 'num_emails': 1, 'syntax_avg': 0, 'syntax_moment': 0, 'syntax_std': 0}
            for intent, result in parsed_email['intents'].items():
                meta[intent] = result['id']
            json_obj = json.dumps(meta, indent=4, sort_keys=False)
            
            with open(sender_metadata, 'w') as meta_out:
                meta_out.write(json_obj)
        else:
            new_sketch = False
            with open(sender_metadata, 'r') as meta_in:
                meta = json.load(meta_in)
            n = meta.get('num_emails', 0) + 1
            meta['num_emails'] = n
            for intent, result in parsed_email['intents'].items():
                meta[intent] = min(result['id'], meta[intent])
            old_mu = meta['syntax_avg']
            y = _compare_syntax(parsed_email['body'], syntax_sketch)
            new_mu = old_mu + (y - old_mu) / (n - 1)
            old_moment = meta['syntax_moment']
            new_moment = old_moment + (y - old_mu) * (y - new_mu)
            meta['syntax_avg'] = new_mu
            meta['syntax_moment'] = new_moment
            meta['syntax_std'] = new_moment / (n-2) if n > 2 else 0
            json_obj = json.dumps(meta, indent=4, sort_keys=False)
            with open(sender_metadata, 'w') as meta_out:
                meta_out.write(json_obj)
        
        with open(syntax_sketch, 'a', newline='') as syntax_csv:
            writer = csv.DictWriter(syntax_csv, fieldnames=SYNTAX_HEADER)
            if new_sketch:
                writer.writeheader()
            rows = _dictorize_email(parsed_email['body'])
            for row in rows:
                writer.writerow(row)

        with open(time_sketch, 'a', newline='') as time_csv:
            writer = csv.DictWriter(time_csv, fieldnames=TIME_HEADER)
            if new_sketch:
                writer.writeheader()
            writer.writerow({'time': parsed_email['arrival_time'].timestamp()})
        
        if not new_sketch:
            clean_up_syntax(syntax_sketch)
  

def clean_up_syntax(sender):
    """
    For pre-existing sketches, load in the syntax counts and aggregate the new tokens with any
    already seen tokens.

    Parameters
    ----------
    sender: str
        The sender syntax file to update.
    """
    if isinstance(sender, dict):
        sender = sender['sender']
    if isinstance(sender, str):
        if os.path.isfile(sender):
            syntax_sketch = sender
        else:
            base_file = hashlib.sha256(sender.encode('utf8')).hexdigest()
            syntax_sketch = os.path.join(SKETCHES, base_file+'_syntax.csv')
    syntax_df = pd.read_csv(syntax_sketch)
    updated = syntax_df.groupby(['token']).sum().reset_index()
    updated.to_csv(syntax_sketch, index=False)
