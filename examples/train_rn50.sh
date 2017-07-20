#!/bin/bash

if [ "x$DATA_TRAIN_PATH" == x ]; then
  read -p "training dataset path (incl. file name): " DATA_TRAIN_PATH
fi

if [ "x$DATA_VALID_PATH" == x ]; then
  read -p "validation dataset path (incl. file name): " DATA_VALID_PATH
fi

if [ "x$USE_NUM_GPUS" == x ]; then
  read -p "How many GPUs to use: " USE_NUM_GPUS
fi

if [ "x$BATCH_PER_GPU" == x ]; then
  read -p "Batch size per GPU: " BATCH_PER_GPU
fi


if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; 
  then MAX_GPUS=$(nvidia-smi -L | wc -l);
  else MAX_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | awk '{print NF}') 
fi

if [ $USE_NUM_GPUS -gt $MAX_GPUS ]; then
   echo -e "Warning: You requested $USE_NUM_GPU but the max. on this system is $MAX_GPUS"
   USE_NUM_GPUS=$MAX_GPUS
fi

python resnet50.py \
  --data-train ${DATA_TRAIN_PATH} \
  --data-val ${DATA_VALID_PATH} \
  --num-gpus ${USE_NUM_GPUS} \
  --batch-per-gpu ${BATCH_PER_GPU}

