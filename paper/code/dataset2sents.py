#!/usr/bin/env python

import csv
import nltk
import sys

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# read in corpus
corpus_filename = sys.argv[1]

with open(corpus_filename, 'r') as csvfile:
	corpus_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in corpus_reader:
		excerpt = row[-1].strip()
		label = row[0]

		for sent in sent_detector.tokenize(excerpt):
			print label + "\t" + sent
