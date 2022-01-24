#!/bin/bash
DATASET=$1
LAMBDA=$2
LAMBDA_1=$3
LAMBDA_2=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

execProg="./optimization_ILP_LPRnd.py"
evalProg="./eval_single.py"

cd ./src

python "$execProg" \
       --dataset ${DATASET} \
       --lam ${LAMBDA} \
       --lam1 ${LAMBDA_1} \
       --lam2 ${LAMBDA_2} \
       ${EXTRA_ARGS}

python "$execProg" \
       --dataset ${DATASET} \
       --lam 1.0 \
       --lam1 0.0 \
       --lam2 0.0 \
       ${EXTRA_ARGS}

python "$execProg" \
       --dataset ${DATASET} \
       --lam 0.0 \
       --lam1 1.0 \
       --lam2 0.0 \
       ${EXTRA_ARGS}

python "$execProg" \
       --dataset ${DATASET} \
       --lam 0.0 \
       --lam1 0.0 \
       --lam2 1.0 \
       ${EXTRA_ARGS}

python "$evalProg" \
       --dataset ${DATASET} \
       --lam ${LAMBDA} \
       --lam1 ${LAMBDA_1} \
       --lam2 ${LAMBDA_2} \
       ${EXTRA_ARGS}