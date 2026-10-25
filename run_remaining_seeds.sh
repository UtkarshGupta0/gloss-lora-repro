#!/bin/bash
RESULTS_FILE="multiseed_results.csv"
echo "loss,seed,test_acc,test_f1" > $RESULTS_FILE
echo "ce,42,0.9758,0.9370" >> $RESULTS_FILE
echo "gloss,42,0.9740,0.9235" >> $RESULTS_FILE

for seed in 7 123; do
  for loss_type in ce gloss; do
    EXTRA_ARGS=""
    [ "$loss_type" == "gloss" ] && EXTRA_ARGS="--sigmafn root --lam 0.8 --gamma 0.6"
    echo "=== Running loss=$loss_type seed=$seed ==="
    LOG=$(python main.py --dataset R8 --loss $loss_type $EXTRA_ARGS --bert_init distilbert-base-uncased --bert_lr 2e-5 --batch_size 16 --max_length 128 --seed $seed 2>&1)
    echo "$LOG" > "run_${loss_type}_seed${seed}.log"
    ACC=$(echo "$LOG" | grep "Test Accuracy" | tail -1 | awk '{print $NF}')
    F1=$(echo "$LOG" | grep "Test F1 score" | tail -1 | awk '{print $NF}')
    echo "$loss_type,$seed,$ACC,$F1" >> $RESULTS_FILE
    echo "=== Done: acc=$ACC f1=$F1 ==="
  done
done
echo "ALL RUNS COMPLETE"
