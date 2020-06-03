#!/bin/bash
for i in SG SA FC NF
do
	for j in 8 9 10 11 12
	do
		csplit $i-$j.txt --suppress-matched 201
		rm $i-$j.txt
		mv xx00 $i-$j-train.txt
		mv xx01 $i-$j-test.txt
	done
done
