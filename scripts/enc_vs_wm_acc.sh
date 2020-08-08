for i in SG SA FC NF
do
    for ((j=0; j<10; j++))
    do
        #echo "./indirection_model_w_enc-dec.py data/$i-10-train.txt data/$i-10-test.txt | tail -2 >> *_wm_acc_${i}_2.txt"
        #echo "./nested-enc-dec.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt | tail -3 >> *_nested_acc_${i}_2.txt"
        echo "./e2e_enc_dec.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt | tail -2 >> \*_e2e_enc_${i}.txt"
        echo "./e2e_enc_dec_query.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt 1 | tail -2 >> \*_e2e_enc_query_1_${i}.txt"
        echo "./e2e_enc_dec_query.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt 5 | tail -2 >> \*_e2e_enc_query_5_${i}.txt"
        echo "./e2e_enc_dec_query.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt 10 | tail -2 >> \*_e2e_enc_query_10_${i}.txt"
    done
done