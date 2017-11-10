import csv

# Opens CSV file and reads it in as a dictionary
with open('vague_sentences_groups_of_5_with_ids.csv') as f:
    turk_data = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

d = {}
for row in turk_data:
	for i in range(1,6):
		if row['sentenceid' + str(i)] == None:
			continue
		id = int(row['sentenceid' + str(i)])
		if d.has_key(id):
			d[id] += 1
		else:
			d[id] = 1

for id, count in d.items():
	if count > 1:
		print id

print 'done'