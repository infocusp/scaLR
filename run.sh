#! /bin/bash

# Feature chunking
# /home/saiyam/anaconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/extract_features.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log

# Training
/home/saiyam/anaconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/train.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log

# Testing
/home/saiyam/anaconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/eval.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log