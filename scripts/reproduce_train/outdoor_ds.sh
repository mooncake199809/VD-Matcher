#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=840
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"

n_nodes=2
n_gpus_per_node=8
torch_num_workers=8
batch_size=2
pin_memory=true
exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO srun -p pat_taurus -x SH-IDC1-10-142-5-144 -N 2 -n 16 --gres gpu:8 python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=40
