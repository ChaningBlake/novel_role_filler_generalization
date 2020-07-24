for i in SG SA FC NF
do
    for ((j=0; j<10; j++))
    do
        #echo "./indirection_model_w_enc-dec.py data/$i-10-train.txt data/$i-10-test.txt | tail -2 >> *_wm_acc_${i}_2.txt"
        echo "./nested-enc-dec.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt | tail -3 >> *_nested_acc_${i}_2.txt"
    done
done