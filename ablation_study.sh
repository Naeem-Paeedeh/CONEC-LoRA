#!/bin/bash

DEVICE=0

for ORDER in {1..5}; do
    python main.py ./exps/ablation/cddb,Cosine_classifier_instead_of_stochastic_classifier.json -order $ORDER -device $DEVICE
    python main.py ./exps/ablation/cddb,Linear_classifier_instead_of_stochastic_classifier.json -order $ORDER -device $DEVICE
    python main.py ./exps/ablation/cddb,without_shared_LoRAs.json -order $ORDER -device $DEVICE
    python main.py ./exps/ablation/cddb,without_intermediate_domain_classifiers.json -order $ORDER -device $DEVICE
    python main.py ./exps/ablation/cddb,without_generator_loss.json -order $ORDER -device $DEVICE
done

