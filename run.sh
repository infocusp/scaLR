#! /bin/bash

# Feature chunking
# /home/saiyam/miniconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/extract_features.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log

# Training
/home/saiyam/miniconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/train.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log

# Testing
/home/saiyam/miniconda3/envs/bioc/bin/python /home/saiyam/projects/biocusp/scripts/eval.py --config /home/saiyam/projects/biocusp/scripts/config.yml --log