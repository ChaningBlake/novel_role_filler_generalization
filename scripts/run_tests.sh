for i in SG SA FC NF
do
	for j in 8 9 10 11 12
	do
		echo "./word-enc-dec.py $i-$j-train.txt $i-$j-train.txt 6 | tail -n +5 > $i-$j-train.data"
		echo "./word-enc-dec.py $i-$j-train.txt $i-$j-test.txt 6 | tail -n +5 > $i-$j-test.data"
	done
done
