#!/bin/bash

#cd ...

# custom config
DATA="DATA/"
MODEL=cluster # "model of aggregation, choose from:cluster(used with CAPT), fedavg, fedprox, local(The last three are used with PromptFL)
TRAINER=CAPT  # name of trainer, choose from: CLIP, PromptFL, CAPT, CoOp, CoCoOp, MaPLe, FedClip, ClipLora, KgCoOp
PRETRAINED=True
IID=False

LR=0.001
GAMMA=1
USERS=20
FRAC=0.4
ROUND=100

CFG=vit_b16  # config file  vit_b16 or rn50
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
N_general=1
CTXINIT=False
CSC=True  # class-specific context (False or True)

SEED=1
BATCH_SIZE=32
TEST_BATCH_SIZE=100
#BETA=0.5
IMB_FACTOR=0.01
SIMCLUST=4
DISCLUSTERS=4
PARTITION=noniid-labeldir

for DATASET in cifar10_LT
do
  # Set PARTITION based on DATASET
  case "$DATASET" in
  cifar10_LT|fmnist_LT)
    NUM_CLASSES=10
    ;;
  cifar100_LT)
    NUM_CLASSES=100
    ;;
  imagenet_LT)
    NUM_CLASSES=1000
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    continue
    ;;
  esac
  for BETA in 0.5
  do
     DIR=output/${DATASET}/${TRAINER}_${MODEL}_${CFG}_batchSize${BATCH_SIZE}/IF=${IMB_FACTOR}_beta=${BETA}/
      if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
      CUDA_VISIBLE_DEVICES=0 python federated_main.py \
      --root ${DATA} \
      --model ${MODEL} \
      --dataset ${DATASET} \
      --seed ${SEED} \
      --num_users ${USERS} \
      --frac ${FRAC} \
      --lr ${LR} \
      --csc ${CSC} \
      --gamma ${GAMMA} \
      --trainer ${TRAINER} \
      --round ${ROUND} \
      --partition ${PARTITION} \
      --beta ${BETA} \
      --n_ctx ${NCTX} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CAPT/${CFG}.yaml \
      --output-dir ${DIR} \
      --imb_factor ${IMB_FACTOR} \
      --ctx_init ${CTXINIT} \
      --train_batch_size ${BATCH_SIZE} \
      --test_batch_size ${TEST_BATCH_SIZE} \
      --num_classes ${NUM_CLASSES} \
      --n_general ${N_general} \
      --n_simclusters ${SIMCLUST} \
      --n_disclusters ${DISCLUSTERS}
    fi
  done
done