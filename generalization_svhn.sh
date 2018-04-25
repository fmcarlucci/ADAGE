#!/usr/bin/env bash
python new_main.py --epoch 800 --source mnist_m mnist synth --target svhn --data_aug_mode simple --source_limit 20000 --target_limit 20000 --use_deco --generalization --classifier multi $1
