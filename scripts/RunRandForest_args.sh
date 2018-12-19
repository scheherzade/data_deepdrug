#!/bin/bash
dir="/home/sshirzad/workspace/deepdrug"
date_str='2018-12-12-1812'
mkdir -p "$dir/results/$date_str"
i=$1
touch $dir/results/$date_str/results_RF_${date_str}_$((i)).txt
ipython ${dir}/python_scripts/randomforest_args.py $((i)) | tee $dir/results/$date_str/results_RF_${date_str}_$((i)).txt
