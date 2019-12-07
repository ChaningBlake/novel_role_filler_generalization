#!/bin/bash
for i in SG SA FC NF
do
	for j in 8 9 10 11 12
	do
		cat $i-$j-pre.txt | ./combo2letters.py 200 100 > $i-$j.txt
	done
done
