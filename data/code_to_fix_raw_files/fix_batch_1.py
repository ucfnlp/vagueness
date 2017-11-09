import numpy as np
import csv

csv_file = 'batch_0_999.csv'
sentences_file = 'vague_sentences_groups_of_5_with_ids.csv'
output_file = 'batch_0_999_fixed.csv'

print ('Reading file: ' + csv_file)

# Opens CSV file and reads it in as a dictionary
with open(csv_file) as f:
    turk_data = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

with open(sentences_file) as f:
    sentences = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    sentences = sentences[:200]


def findIds(target_sentence):
	res = []
	for sent in sentences:
		for j in range(5):
			if sent['sentence' +str(j+1)] == target_sentence:
				res.append((int(sent['sentenceid' + str(j+1)]), int(sent['docid' + str(j+1)])))
	return res

new_data = []
for row in turk_data:
	new_row = row.copy()
	possible_ids = findIds(row['Input.sentence'])
	sentence_ids = [x[0] for x in possible_ids]
	doc_ids = [x[1] for x in possible_ids]
	if len(doc_ids) == 1:
		new_row['Input.docid'] = doc_ids[0]
	else:
		new_row['Input.docid'] = doc_ids
	if len(doc_ids) == 1:
		new_row['Input.sentenceid'] = sentence_ids[0]
	else:
		new_row['Input.sentenceid'] = sentence_ids
	new_data.append(new_row)

def hasBeenUsedAlready(sentenceid):
	for i, row in enumerate(new_data):
		if type(row['Input.sentenceid']) == list:
			continue
		if row['Input.sentenceid'] == sentenceid:
			print 'sentenceid ' + str(sentenceid) + ' was used at ' + str(i)
			return True
	print 'sentenceid ' + str(sentenceid) + ' has not been used already'
	return False

groupDocId = None
groupSentenceId = None
for row_idx, row in enumerate(new_data):
	print 'row ' + str(row_idx)
	if groupSentenceId:
		if type(row['Input.sentenceid']) == int or (not groupSentenceId in row['Input.sentenceid']):
			groupDocId = None
			groupSentenceId = None
		else:
			row['Input.docid'] = groupDocId
			row['Input.sentenceid'] = groupSentenceId
	else:
		if type(row['Input.sentenceid']) == int:
			continue
		else:
			foundOne = False
			for i in range(len(row['Input.sentenceid'])):
				if hasBeenUsedAlready(row['Input.sentenceid'][i]):
					continue
				else:
					foundOne = True
					row['Input.docid'] = row['Input.docid'][i]
					row['Input.sentenceid'] = row['Input.sentenceid'][i]
					groupDocId = row['Input.docid']
					groupSentenceId = row['Input.sentenceid']
					break
			if not foundOne:
				raise Exception("Didnt find one for " + str(row_idx))



with open(output_file, 'wb') as f:
	w = csv.DictWriter(f,new_data[0].keys())
	w.writeheader()
	w.writerows(new_data)