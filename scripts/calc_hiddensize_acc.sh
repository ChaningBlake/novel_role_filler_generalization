for i in SG SA FC NF
do
	echo "-----$i-----" >> layer_accuracy.txt
	for j in 2 4 8 10
	do
		echo -n "12w-${j}hl-train: " >> layer_accuracy.txt 
		./accuracy.py $i-12-train_SIZE$j.data $i-12-train.txt >> layer_accuracy.txt
		echo -n "12w-${j}hl-test: " >> layer_accuracy.txt 
		./accuracy.py $i-12-test_SIZE$j.data $i-12-test.txt >> layer_accuracy.txt
	done
done
