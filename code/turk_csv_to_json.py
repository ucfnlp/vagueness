#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json

# The file we read our annotated data from
csv_file = '../data/Batch_0_999_approval_7_sorted.csv'

# We output to this file the document IDs for each sentence in the dataset
vague_sents_doc_file = '../data/vague_sents_doc.txt'

# The file we output our clean JSON data to.
clean_data_json = '../data/clean_data.json'

# Represents a privacy policy document, containing many sentences
class Document(object):

    def __init__(self):
        self.type = 'Document'
        self.id = None
        
        # List of the Sentence objects in this document
        self.vague_sentences = []
        
    # Used to convert to JSON
    def reprJSON(self):
        return dict(type=self.type, id=self.id, vague_sentences=self.vague_sentences) 
        
# Represents a sentence, and holds info and annotations about this sentence
class Sentence(object):

    def __init__(self):
        self.type = 'Sentence'
        
        # Amazon Mechanical Turk's ID for this sentence
        self.hit_id = None
        
        # Plain string representation of the sentence
        self.sentence_str = None
        
        # List of the sentence-level vagueness scores (up to 5, 1 for each annotator)
        self.scores = []
        
        # Histogram of the vague phrases for this sentence, and the # of annotators that entered it
        self.vague_phrases = {}
        
    # Used to convert to JSON
    def reprJSON(self):
        return dict(type=self.type, hit_id=self.hit_id, sentence_str=self.sentence_str, 
                    scores=self.scores, vague_phrases=self.vague_phrases) 
        
# Encodes objects into JSON
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)
        
# Opens CSV file and reads it in as a dictionary
with open(csv_file) as f:
    turk_data = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    
# Reads in the document ID for each sentence
with open(vague_sents_doc_file) as f:
    vague_doc_inidices = f.read().splitlines()
    
docs = []
sentence_idx = 0
cur_doc = Document()
cur_sent = Sentence()


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

print('done')










































