import numpy as np
import csv

csv_files = ['batch_1000_1499.csv','batch_1500_1999.csv']
sentences_file = 'vague_sentences_groups_of_5_with_ids.csv'
output_files = ['batch_1000_1499_fixed.csv','batch_1500_1999_fixed.csv']

def fixFile(csv_file, sentences, output_file):

	print ('Reading file: ' + csv_file)

	# Opens CSV file and reads it in as a dictionary
	with open(csv_file) as f:
	    turk_data = [{k: v for k, v in row.items()}
	        for row in csv.DictReader(f, skipinitialspace=True)]


	def findIds(target_sentence):
		res = []
		for i, sent in enumerate(sentences):
			for j in range(5):
				if sent['sentence' +str(j+1)] == target_sentence:
					res.append((int(sent['sentenceid' + str(j+1)]), int(sent['docid' + str(j+1)])))
					if i == 31550:
						print 
		return res

	new_data = []
	for row in turk_data:
		new_row = row.copy()
		new_data.append(new_row)
		for idx in range(1,6):
			possible_ids = findIds(row['Input.sentence' + str(idx)])
			sentence_ids = [x[0] for x in possible_ids]
			doc_ids = [x[1] for x in possible_ids]
			if len(doc_ids) == 1:
				new_row['Input.docid' + str(idx)] = doc_ids[0]
			else:
				new_row['Input.docid' + str(idx)] = doc_ids
			if len(doc_ids) == 1:
				new_row['Input.sentenceid' + str(idx)] = sentence_ids[0]
			else:
				new_row['Input.sentenceid' + str(idx)] = sentence_ids

	def hasBeenUsedAlready(sentenceid):
		for i, row in enumerate(new_data):
			for j in range(1,6):
				sid = 'Input.sentenceid' + str(j)
				if type(row[sid]) == list:
					continue
				if row[sid] == sentenceid:
					print 'sentenceid ' + str(sentenceid) + ' was used at ' + str(i) + ' idx ' + str(j)
					return True
		print 'sentenceid ' + str(sentenceid) + ' has not been used already'
		return False

	for j in range(1,6):
		sid = 'Input.sentenceid' + str(j)
		did = 'Input.docid' + str(j)
		groupDocId = None
		groupSentenceId = None
		for row_idx, row in enumerate(new_data):
			print 'row ' + str(row_idx)
			if groupSentenceId:
				if type(row[sid]) == int or (not groupSentenceId in row[sid]):
					groupDocId = None
					groupSentenceId = None
				else:
					row[did] = groupDocId
					row[sid] = groupSentenceId
					continue
			if type(row[sid]) == int:
				continue
			else:
				foundOne = False
				for i in range(len(row[sid])):
					if type(row[sid][i]) != int:
						raise Exception(str(row[sid][i]) + ' is not an int')
					if hasBeenUsedAlready(row[sid][i]):
						continue
					else:
						foundOne = True
						row[did] = row[did][i]
						row[sid] = row[sid][i]
						print row[sid]
						groupDocId = row[did]
						groupSentenceId = row[sid]
						break
				if not foundOne:
					raise Exception("Didnt find one for " + str(row_idx) + ' idx ' + str(j))
				if type(row[sid]) == list:
					raise Exception(str(row_idx) + ' idx ' + str(j) + ' is still a list')

	for row_idx, row in enumerate(new_data):
		for j in range(1,6):
			sid = 'Input.sentenceid' + str(j)
			if type(row[sid]) == list:
				raise Exception(str(row_idx) + ' idx ' + str(j) + ' is still a list')

	with open(output_file, 'wb') as f:
		w = csv.DictWriter(f,new_data[0].keys())
		w.writeheader()
		w.writerows(new_data)



with open(sentences_file) as f:
    sentences = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    sentences2 = sentences[200:300]
    sentences3 = sentences[300:400]
sentences_list = [sentences2, sentences3]

for i in range(len(sentences_list)):
	print 'Fixing file' + csv_files[i]
	fixFile(csv_files[i], sentences_list[i], output_files[i])