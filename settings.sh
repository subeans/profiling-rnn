#!/bin/bash
# Based on AWS DeepLearning AMI with Ubuntu 18.04

rm /usr/local/cuda
ln -s /usr/local/cuda-11.0 /usr/local/cuda

# Upgrade pip3 and install tensorflow, tensorboard, dataset packages
pip3 install --upgrade pip
pip3 install tensorflow
# pip3 install tf-nightly
# pip3 install tb-nightly
pip3 install tensorboard-plugin-profile
pip3 install tensorflow_datasets
# for bert model 
pip3 install tesorflow-text
pip3 install tensorflow-hub
pip3 install tf-models-official
mkdir tensorstats
