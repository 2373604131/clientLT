@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."

REM Windows test preset for CAPT on CIFAR100-LT with a 16GB GPU.
REM Increase ROUND and BATCH_SIZE after this smoke test runs successfully.
set "DATA=DATA/"
set "MODEL=cluster"
set "TRAINER=CAPT"
set "DATASET=cifar100_LT"
set "NUM_CLASSES=100"

set "LR=0.001"
set "GAMMA=1"
set "USERS=20"
set "FRAC=0.4"
set "ROUND=100"

set "CFG=vit_b16"
set "NCTX=4"
set "N_GENERAL=1"
set "CTXINIT=False"
set "CSC=True"

set "SEED=1"
set "BATCH_SIZE=32"
set "TEST_BATCH_SIZE=64"
set "NUM_WORKERS=4"
set "GLOBAL_EVAL_INTERVAL=5"
set "IMB_FACTOR=0.01"
set "IMB_TYPE=exp"
set "SIMCLUST=4"
set "DISCLUSTERS=4"
set "PARTITION=noniid-labeldir"
set "CUDA_VISIBLE_DEVICES=0"

for %%B in (0.5) do (
  set "BETA=%%B"
  set "DIR=output/%DATASET%/%TRAINER%_%MODEL%_%CFG%_batchSize%BATCH_SIZE%/IF=%IMB_FACTOR%_beta=%%B_partition=%PARTITION%/"

  echo =========================================================
  echo Running CAPT on %DATASET%
  echo Output dir: !DIR!
  echo =========================================================

  python federated_main.py ^
    --root "%DATA%" ^
    --model "%MODEL%" ^
    --dataset "%DATASET%" ^
    --seed "%SEED%" ^
    --num_users "%USERS%" ^
    --frac "%FRAC%" ^
    --lr "%LR%" ^
    --csc "%CSC%" ^
    --gamma "%GAMMA%" ^
    --trainer "%TRAINER%" ^
    --round "%ROUND%" ^
    --partition "%PARTITION%" ^
    --beta "!BETA!" ^
    --n_ctx "%NCTX%" ^
    --dataset-config-file "configs/datasets/%DATASET%.yaml" ^
    --config-file "configs/trainers/CAPT/%CFG%.yaml" ^
    --output-dir "!DIR!" ^
    --imb_factor "%IMB_FACTOR%" ^
    --imb_type "%IMB_TYPE%" ^
    --ctx_init "%CTXINIT%" ^
    --train_batch_size "%BATCH_SIZE%" ^
    --test_batch_size "%TEST_BATCH_SIZE%" ^
    --global_eval_interval "%GLOBAL_EVAL_INTERVAL%" ^
    --num_classes "%NUM_CLASSES%" ^
    --n_general "%N_GENERAL%" ^
    --n_simclusters "%SIMCLUST%" ^
    --n_disclusters "%DISCLUSTERS%" ^
    DATALOADER.NUM_WORKERS "%NUM_WORKERS%"

  if errorlevel 1 (
    echo CAPT run failed.
    popd
    exit /b 1
  )
)

popd
endlocal
