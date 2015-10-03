#!/usr/bin/env python
# generate document frequencies for tf.idf

def preproc(in_string):
	in_string = in_string.replace("'", " ' ")
	return in_string

import csv
import nltk
import operator
import sys

# read in corpus
corpus_filename = sys.argv[1]

labels = []
docs = []
corpus = ''
with open(corpus_filename, 'r') as csvfile:
	corpus_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in corpus_reader:
		excerpt = row[-1].strip()

		excerpt = preproc(excerpt)

		corpus += excerpt + ' '
		docs.append(excerpt)

		label = row[0]
		labels.append(label)

# break corpus into words, build a list of those
words = set([])
words_lower = set([])

df = {}
df_lower = {}

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
for sent in sent_detector.tokenize(corpus):
	for word in nltk.word_tokenize(sent):
		words.add(word)
		words_lower.add(word.lower())

# build a dict of #docs per word
for doc in docs:
	doc_words = set([])
	doc_words_lower = set([])
	for sent in sent_detector.tokenize(doc):
		for word in nltk.word_tokenize(sent):
			doc_words.add(word)
			doc_words_lower.add(word.lower())

	for word in doc_words:
		if word not in df:
			df[word] = 0
		df[word] += 1
	for word in doc_words_lower:
		if word not in df_lower:
			df_lower[word] = 0
		df_lower[word] += 1


# print top terms (on tf.idf) per doc
for i in range(len(docs)):
	doc = docs[i]
	tfs = {}
	for sent in sent_detector.tokenize(doc):
		for word in nltk.word_tokenize(sent):
			word = word.lower()
			if word not in tfs:
				tfs[word] = 0
			tfs[word] += 1

	tfidfs = {}
	for term in tfs:
		tfidfs[term] = float(tfs[term]) / df_lower[term]

	sorted_tfidfs = sorted(tfidfs.items(), key=operator.itemgetter(1), reverse=True)

	crf_feats = []
	for sorted_tfidf in sorted_tfidfs:
		crf_feats.append(sorted_tfidf[0] + ':' + str(sorted_tfidf[1]))
	
	for label in labels[i].split(','):
		print
		if not label.strip():
			label = 'none'
		print label + "\t" + "\t".join(crf_feats)

