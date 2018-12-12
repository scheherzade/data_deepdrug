#!/bin/bash
python_script='CNN_11_13.py'
date_str='2018-11-13-1221'
curr_date=`date '+%Y-%m-%d-%H%M'`
results_dir="/home/sshirzad/workspace/deepdrug/results/$date_str/$curr_date"
mkdir -p ${results_dir}
cd ${results_dir}
ipython /home/sshirzad/workspace/deepdrug/python_scripts/${python_script} | tee ${results_dir}_1
cp /home/sshirzad/workspace/deepdrug/python_scripts/${python_script} ${results_dir}

