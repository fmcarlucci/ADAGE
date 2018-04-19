#!/usr/bin/env bash
python new_main.py --epoch 200 --source mnist svhn synth --target mnist_m --data_aug_mode simple --source_limit 20000 --use_deco --generalization --classifier multi --deco_blocks 8