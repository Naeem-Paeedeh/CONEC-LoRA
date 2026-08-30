#!/bin/bash

DEVICE=0

DATASETS="cddb domainnet officehome core50"
# DATASETS="cddb"

for DS in $DATASETS; do
    for ORDER in {1..5}; do
        python main.py ./exps/main_experiments/${DS}.json -order $ORDER -device $DEVICE
    done
done
