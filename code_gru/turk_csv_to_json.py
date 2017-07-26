#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import os

# The file we read our annotated data from
csv_folder = '../data/raw_csv_files'

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
        
        # Sentence id
        self.id = None
        
        # Plain string representation of the sentence
        self.sentence_str = None
        
        # List of the sentence-level vagueness scores (up to 5, 1 for each annotator)
        self.scores = []
        
        # Histogram of the vague phrases for this sentence, and the # of annotators that entered it
        self.vague_phrases = {}
        
        # Document id (just for internal use)
        self.doc_id = None
        
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

docs = {}        
for fn in os.listdir(csv_folder):
     if os.path.isfile(csv_folder+'/'+fn) and fn.endswith('.csv'):
        print ('Reading file:', fn)
        
        # Opens CSV file and reads it in as a dictionary
        with open(csv_folder+'/'+fn) as f:
            turk_data = [{k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]
        if not turk_data[0].has_key('Input.docid1'):
            print ('skipping file because it does not include document id and sentence id')
            continue
#             # Reads in the document ID for each sentence
#             with open(vague_sents_doc_file) as f:
#                 vague_doc_inidices = f.read().splitlines()
            
        cur_doc = Document()
        sentences = {}
        
        # for each hit in csv
        for i in range(len(turk_data)):
            # for each of the 5 sentences in this hit
            for j in range(1,6):
                idx = str(j)
                row = turk_data[i]
                sent_id = row['Input.sentenceid'+idx]
                if sentences.has_key(sent_id):
                    cur_sent = sentences[sent_id]
                else:
                    cur_sent = Sentence()
                    cur_sent.hit_id = row['HITId']
                    cur_sent.doc_id = row['Input.docid'+idx]
                    cur_sent.sentence_str = row['Input.sentence'+idx]
                    sentences[sent_id] = cur_sent
                    
                if(row['AssignmentStatus'] == 'Approved'):
                    cur_sent.scores.append(row['Answer.score'+idx])
                    phrases_str = row['Answer.words'+idx]
                    if phrases_str != '{}':
                        phrases = [x.strip().lower() for x in phrases_str.split(',')]
                        for phrase in phrases:
                            cur_sent.vague_phrases[phrase] = cur_sent.vague_phrases.get(phrase, 0) + 1

        
        for sent in sentences.values():
            doc_id = sent.doc_id
            if not docs.has_key(doc_id):
                new_doc = Document()
                new_doc.id = doc_id
                docs[doc_id] = new_doc
            docs[doc_id].vague_sentences.append(sent)
        
        
        
#         for i in range(len(turk_data)):
#             cur_doc.id = vague_doc_inidices[sentence_idx]
#             row = turk_data[i]
#             cur_sent.hit_id = row['HITId']
#             cur_sent.sentence_str = row['Input.sentence']
#             if(row['AssignmentStatus'] == 'Approved'):
#                 cur_sent.scores.append(row['Answer.score'])
#                 phrases_str = row['Answer.words']
#                 if phrases_str != '{}':
#                     phrases = [x.strip().lower() for x in phrases_str.split(',')]
#                     for phrase in phrases:
#                         cur_sent.vague_phrases[phrase] = cur_sent.vague_phrases.get(phrase, 0) + 1
#             
#             
#             if i+1 >= len(turk_data):
#                 cur_doc.vague_sentences.append(cur_sent)
#                 docs.append(cur_doc)
#             else:
#                 next_row = turk_data[i+1]
#                 if next_row['HITId'] != cur_sent.hit_id:
#                     cur_doc.vague_sentences.append(cur_sent)
#                     cur_sent = Sentence()
#                     sentence_idx += 1
#                 if vague_doc_inidices[sentence_idx] != cur_doc.id:
#                     docs.append(cur_doc)
#                     cur_doc = Document()

docs = docs.values()
with open(clean_data_json, 'w') as f:
    f.write(json.dumps({'docs': docs}, cls=ComplexEncoder, indent=4, ))

print('done')










































