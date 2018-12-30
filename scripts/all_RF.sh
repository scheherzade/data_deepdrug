#!/bin/bash
for i in $(seq 7)
do 
echo $((i+17))
sbatch -p marv_ht  -N 1 --time=72:00:00 /home/sshirzad/workspace/deepdrug/scripts/RunRF.sh $((i+17))
done

