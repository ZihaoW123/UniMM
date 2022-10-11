#!/usr/bin/env bash

mkdir checkpoints-release

# VQA ViLBERT pretrained weights
wget https://s3.amazonaws.com/visdial-bert/checkpoints/vqa_weights -O checkpoints-release/vqa_pretrained_weights
