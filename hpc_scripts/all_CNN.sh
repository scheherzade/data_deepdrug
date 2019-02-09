#!/bin/bash
for i in $(seq 1)
do 
test_id=$((i-1))
echo $test_id
qsub /home/sshirz1/runs/scripts/RunCNN.sh -F $test_id -q gpu -o "CNN_${test_id}_o" -e "CNN_${test_id}_e"
#sbatch -p bahram  -N 1 --time=72:00:00 /home/sshirzad/workspace/deepdrug/scripts/RunCNN.sh $((i-1))
done


