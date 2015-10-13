#!/usr/bin/env python3
# params: w2v model file, labeled sentences

import nltk
import sys
from word2vec import Word2Vec, Sent2Vec, LineSentence

w2v_model_file = sys.argv[1]
sent_file = sys.argv[2]

global_labels = set([])
y = []
X_text = []

print('reading in dataset')
for line in open(sent_file):
	if not line.strip():
		continue
	label_string, text = line.rstrip().split("\t") # rstrip instead of strip, because precending tabs are important here 
	labels = label_string.split(',')
	y.append(labels)
	X_text.append(' '.join(nltk.word_tokenize(text)))
	if labels:
		for label in labels:
			global_labels.add(label)

# generate sentence vectors
print('building sentence vectors')
X = []
sentence_model = Sent2Vec(X_text, model_file=w2v_model_file)
for i in range(len(X_text)):
	# convert from numpy array
	representation = list(val for val in sentence_model.sents[i])
	X.append(representation)

# generate an svmlight file for each label
print('writing feature representations')
for label in global_labels:
	if not label:
		continue
	outfile_name = '.'.join([sent_file, label, 'svmlight.features'])
	outfile = open(outfile_name, 'w')
	print ('-> ' , outfile_name)
	positives = 0
	for i in range(len(y)):
		target = -1.0 if label not in y[i] else 1.0
		if target == 1.0:
			positives += 1 
		out_line = str(target)
		for j in range(len(X[i])):
			out_line += ' ' + str(j + 1) + ':' + str(X[i][j]) # j+1 because svmlight insists numbers start at 1

		outfile.write(out_line + "\n") 
	negatives = len(y) - positives
	skew = max([negatives, positives]) / float(len(y))
	print ('positives', positives, 'negatives', negatives, 'skew', skew)
	outfile.close()

