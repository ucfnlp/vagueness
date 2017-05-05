#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json

csv_file = '../data/Batch_0_999_approval_7_sorted.csv'
vague_sents_doc_file = '../data/vague_sents_doc.txt'
clean_data_json = '../data/clean_data.json'

class Document(object):

    def __init__(self):
        self.type = 'Document'
        self.id = None
        self.vague_sentences = []
    def reprJSON(self):
        return dict(type=self.type, id=self.id, vague_sentences=self.vague_sentences) 
        
class Sentence(object):

    def __init__(self):
        self.type = 'Sentence'
        self.hit_id = None
        self.sentence_str = None
        self.scores = []
        self.vague_phrases = {}
    def reprJSON(self):
        return dict(type=self.type, hit_id=self.hit_id, sentence_str=self.sentence_str, 
                    scores=self.scores, vague_phrases=self.vague_phrases) 
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)
        
def object_decoder(obj):
    if 'type' in obj and obj['type'] == 'Document':
        d =  Document()
        d.type = obj['type']
        d.id = obj['id']
        d.vague_sentences = obj['vague_sentences']
    return obj
# class Phrase(object):
# 
#     def __init__(self):
#         self.phrase_str = None
#         self.num_times_counted_as_vague = None
        

with open(csv_file) as f:
    turk_data = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    
with open(vague_sents_doc_file) as f:
    vague_doc_inidices = f.read().splitlines()
    
docs = []
sentence_idx = 0
cur_doc = Document()
cur_sent = Sentence()
# cur_phrase = Phrase()
for i in range(len(turk_data)):
    cur_doc.id = vague_doc_inidices[sentence_idx]
    row = turk_data[i]
    cur_sent.hit_id = row['HITId']
    cur_sent.sentence_str = row['Input.sentence']
    if(row['AssignmentStatus'] == 'Approved'):
        cur_sent.scores.append(row['Answer.score'])
        phrases_str = row['Answer.words']
        if phrases_str != '{}':
            phrases = [x.strip().lower() for x in phrases_str.split(',')]
            for phrase in phrases:
                cur_sent.vague_phrases[phrase] = cur_sent.vague_phrases.get(phrase, 0) + 1
    
    
    if i+1 >= len(turk_data):
#         cur_sent.vague_phrases.append(cur_phrase)
        cur_doc.vague_sentences.append(cur_sent)
        docs.append(cur_doc)
    else:
        next_row = turk_data[i+1]
        if next_row['HITId'] != cur_sent.hit_id:
            cur_doc.vague_sentences.append(cur_sent)
            cur_sent = Sentence()
            sentence_idx += 1
        if vague_doc_inidices[sentence_idx] != cur_doc.id:
            docs.append(cur_doc)
            cur_doc = Document()

with open(clean_data_json, 'w') as f:
    f.write(json.dumps({'docs': docs}, cls=ComplexEncoder, indent=4, ))
c = 0










































