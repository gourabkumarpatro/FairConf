#!/bin/bash
DATASET=$1
LAMBDA=$2
LAMBDA_1=$3
LAMBDA_2=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

cd ./src

prog="./optimization_ILP_LPRnd.py"

python "$prog" \
       --dataset ${DATASET} \
       --lam ${LAMBDA} \
       --lam1 ${LAMBDA_1} \
       --lam2 ${LAMBDA_2} \
       ${EXTRA_ARGS}
