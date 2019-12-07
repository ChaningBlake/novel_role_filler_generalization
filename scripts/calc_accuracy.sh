for i in SG SA FC NF
do
	echo "-----$i-----" >> all_accuracy.txt
	for j in 8 9 10 11 12
	do
		echo -n "$j-train: " >> all_accuracy.txt 
		./accuracy.py $i-$j-train.data $i-$j-train.txt >> all_accuracy.txt
		echo -n "$j-test: " >> all_accuracy.txt 
		./accuracy.py $i-$j-test.data $i-$j-test.txt >> all_accuracy.txt
	done
done
