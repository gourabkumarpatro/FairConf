#!/bin/bash
DATASET=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

execProg="./optimization_ILP_LPRnd.py"
evalProg="./eval_multiple.py"

cd ./src

for lamda_1Loop in 0.0 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
	python "$execProg" \
       --dataset ${DATASET} \
       --lam 1.0 \
       --lam1 ${lamda_1Loop} \
       --lam2 0.5 \
       ${EXTRA_ARGS}
done

for lamda_2Loop in 0.0 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
	python "$execProg" \
       --dataset ${DATASET} \
       --lam 1.0 \
       --lam1 0.5 \
       --lam2 ${lamda_2Loop} \
       ${EXTRA_ARGS}
done

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
       ${EXTRA_ARGS}