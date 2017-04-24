#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

clean_data_json = '../data/clean_data.json'

np.random.seed(123)
train_ratio = 0.8

with open(clean_data_json) as f:
    json_str = f.read()
data = json.loads(json_str)

doc_ids = set()
for doc in data['docs']:
    doc_ids.add(doc['id'])
doc_ids = list(doc_ids)
np.random.shuffle(doc_ids)
train_len = int(train_ratio*len(doc_ids))
train_doc_ids = doc_ids[:train_len]
test_doc_ids = doc_ids[train_len:]



a = 0











































