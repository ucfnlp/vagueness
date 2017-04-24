import numpy
import cPickle
import h5py

Y_vague_predict_file = 'Y_vague_predict_bidir.out'
test_Y_vague_file = 'test_Y_padded_vague_bidir.out'
word_id_seqs_file = 'word_id_seqs.p'
dataset_file = 'dataset.h5'


Y_vague_predict = numpy.genfromtxt(Y_vague_predict_file,delimiter=",")
test_Y_padded_vague = numpy.genfromtxt(test_Y_vague_file,delimiter=",")

# with open(word_id_seqs_file) as f:
#   word_id_seqs = cPickle.load(f)
#   
# a = 0

with h5py.File(dataset_file, 'r') as data_file:
    test_X_padded = data_file['test_X'][:]
    
if not test_X_padded.shape[0] == Y_vague_predict.shape[0]:
    print('test_X_padded len (' + str(test_X_padded.shape[0])
          + ') does not equal prediction length (' + str(Y_vague_predict.shape[0]) + ')')

d = {}
with open("words.dict") as f:
    for line in f:
       (val, key) = line.split()
       d[int(key)] = val

out = ''
for i in range(test_X_padded.shape[0]):
    line = ''
    for j in range(test_X_padded.shape[1]):
        idx = test_X_padded[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
        if test_Y_padded_vague[i][j] == 1:
            line += '*'
        line += ' '
    line += '\n'
    out += line
with open(test_Y_vague_file + '_words.txt', 'w') as f:
    f.write(out)
out = ''
for i in range(test_X_padded.shape[0]):
    line = ''
    for j in range(test_X_padded.shape[1]):
        idx = test_X_padded[i][j]
        if idx == 0:
            continue
        word = d[idx]
        line += word
        if Y_vague_predict[i][j] == 1:
            line += '*'
        line += ' '
    line += '\n'
    out += line
with open(Y_vague_predict_file + '_words.txt', 'w') as f:
    f.write(out)
print('done')






