#!/bin/bash

DEVICE=0

for ORDER in {1..5}; do
    python main.py ./exps/officehome.json -order $ORDER -device $DEVICE
done
