#!/usr/bin/env python

import nltk
import sys

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

for line in open(sys.argv[1], 'r'):
	for sent in sent_detector.tokenize(line.strip()):
		print sent