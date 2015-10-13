#!/usr/bin/env python3

import nltk
import sys

text = open(sys.argv[1], 'r').read()
text = text.replace("\n", " ")
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(' '.join(nltk.word_tokenize(sentence)))

