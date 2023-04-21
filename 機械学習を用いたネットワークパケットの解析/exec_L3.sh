#!/bin/bash
#SBATCH -p cluster_long
#SBATCH -N 1
#SBATCH -c 52
#SBATCH -t 100:00:00

DIR="out_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir $DIR

nohup ./dl_with_optuna_L3.py > ${DIR}/${DIR}_001.log 2>&1 < /dev/null &
sleep 90

for i in `seq -f %03g 2 52`
do
  nohup ./dl_with_optuna_L3.py > ${DIR}/${DIR}_${i}.log 2>&1 < /dev/null &
done

wait