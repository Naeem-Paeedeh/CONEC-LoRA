#!/bin/bash

DEVICE=0

for ORDER in {1..5}; do
    python main.py ./exps/robustness_analysis/cddb,lambda1=3.json -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda1=4.json -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda1=6.json -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda1=7.json -order $ORDER -device $DEVICE

    python main.py ./exps/robustness_analysis/cddb,lambda2=1.json   -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda2=1.5.json -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda2=2.5.json -order $ORDER -device $DEVICE
    python main.py ./exps/robustness_analysis/cddb,lambda2=3.json   -order $ORDER -device $DEVICE
done

for ORDER in {1..5}; do
    python main.py ./exps/sensitivity_analysis/cddb,shared=2,specific=10.json -order $ORDER -device $DEVICE
    python main.py ./exps/sensitivity_analysis/cddb,shared=4,specific=8.json  -order $ORDER -device $DEVICE
    python main.py ./exps/sensitivity_analysis/cddb,shared=8,specific=4.json  -order $ORDER -device $DEVICE
    python main.py ./exps/sensitivity_analysis/cddb,shared=10,specific=2.json -order $ORDER -device $DEVICE
done

