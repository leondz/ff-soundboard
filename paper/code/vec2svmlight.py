#!/usr/bin/env python
# convert sentence2vec sequence to svmlight sequence by adding sequential feature IDs starting at 1

import sys

for line in open(sys.argv[1], 'r'):
	line = line.strip()
	if line:
		i = 1
		repr = []
		for value in line.split():
			repr.append(str(i) + ':' + value)
			i += 1
		print ' '.join(repr)
