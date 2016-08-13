#!/bin/bash
echo '!!! Joining together files from all Martian years...'
#python preprocessing/merge_data.py

echo '!!! Creating features'
#python preprocessing/prepare_dmop.py
#python preprocessing/prepare_evtf.py
#python preprocessing/prepare_saaf.py
#python preprocessing/prepare_ftl.py

echo '!!! Creating datasets'
#python prepare_data1.py
python prepare_data2.py
