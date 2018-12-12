#!/bin/bash
date_str='2018-11-13-1221'
ipython /home/sshirzad/workspace/deepdrug/python_scripts/randomforest.py | tee /home/sshirzad/workspace/deepdrug/results/results_RF_${date_str}.txt

