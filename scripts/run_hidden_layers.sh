for i in SG SA FC NF
do
	for j in 2 4 8 10
	do
		echo "./word-enc-dec.py $i-12-train.txt $i-12-train.txt $j | tail -n +5 > $i-12-train_SIZE$j.data"
		echo "./word-enc-dec.py $i-12-train.txt $i-12-test.txt $j | tail -n +5 > $i-12-test_SIZE$j.data"
	done
done
