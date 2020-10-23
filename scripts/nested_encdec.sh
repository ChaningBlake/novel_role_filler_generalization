for i in SG SA FC NF
do
    for ((j=0; j<10; j++))
    do
        echo "./nested-enc-dec-query.py data/len5_10000-train.txt data/$i-10-train.txt data/$i-10-test.txt 1 >> .nested_query_${i}_${j}.txt"
    done
done