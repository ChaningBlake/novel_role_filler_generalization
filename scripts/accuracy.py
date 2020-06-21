#!/usr/bin/env python3
'''
Friday October 25th
@author: Blake Mullinax

This script takes the output of the decoder and the expected output
as arguments to compute a score of accuracy.
(Calculated based on each individual character)
Files must be of same length
'''

import sys

if len(sys.argv) != 3:
    print()
    print("%s [decoder output] [expected output]"%sys.argv[0])
    print()
    sys.exit(1)

# Open files
try:
    actual = open(sys.argv[1])
except:
    print("%s does not exist"%sys.argv[1])
    sys.exit(1)
try:
    expected = open(sys.argv[2])
except:
    print("%s does not exist"%sys.argv[2])
    sys.exit(1)

correct = 0 # Total number of matched chars
total = 0 # Total number of chars
'''
Here is the fun matching part
If target is 'abc':
abc          3/4
astop        1/4


'''
for w1, w2 in zip(actual, expected):
    w1 = w1.rstrip()
    w2 = w2.rstrip()
    if "stop" in w1:
        w1 = w1[:-4]
        total += 1
        if len(w1) == 3:
            correct += 1
        while "stop" in w1:
            w1 = w1[:-4]
            w1 += "*"
    elif not("stop" in w1) and len(w1) > 3:
        total += 1
        w1 = w1[:3]
    elif not("stop" in w1):
        total += 1
    if len(w1) < 3:
        for i in range(3-len(w1)):
            w1 += "*"
    for i in range(len(w1)):
        if w1[i] == w2[i]:
            correct += 1
        total += 1
print("%d/%d"%(correct, total))
        

actual.close()
expected.close()
