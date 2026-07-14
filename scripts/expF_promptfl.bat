@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0.."

if not defined DATA set "DATA=DATA/"
if not defined GPU set "GPU=0"
if not defined PYTHON_BIN set "PYTHON_BIN=python"
if not defined DRY_RUN set "DRY_RUN=0"

set "MODEL=fedavg"
set "TRAINER=PromptFL"

set "DATASETS=cifar100_LT"
set "CFG=vit_b16"

set "LR=0.001"
set "GAMMA=1"

set "USERS=50"
set "FRAC=0.2"
set "ROUND=100"
set "LOCAL_EPOCHS=5"

set "BATCH_SIZE=32"
set "TEST_BATCH_SIZE=64"
set "NUM_WORKERS=4"

set "GLOBAL_EVAL_INTERVAL=5"
set "UPDATE_RETENTION_INTERVAL=5"

set "NCTX=4"
set "N_GENERAL=1"
set "CTXINIT=False"
set "CSC=True"

set "IMB_FACTOR=0.01"
set "IMB_TYPE=exp"

set "HEAD_CLIENT_RATIO=0.9"
set "TAIL_CLIENT_RATIO=0.1"
set "HEAD_CLASS_RATIO=0.8"
set "TAIL_CLASS_RATIO=0.2"

set "INTRA_GROUP_ALPHA=0.1"
set "HEAD_LEAKAGE_SCALE=3.0"

set "SEEDS=1 2 3"
set "DIRICHLET_BETAS=1.0 0.5 0.3 0.1"
set "CLIENTLT_LAMBDAS=0.0 0.25 0.5 0.75 1.0"

set "ISOLATE_LOCAL_OPTIMIZER_STATE=True"
set "FEDERATED_SINGLE_SCHEDULER_STEP=True"

set "TOTAL_RUNS=27"
set /a PLANNED_RUNS=0

if not exist "federated_main.py" (
  echo federated_main.py not found.
  exit /b 1
)

for %%D in (%DATASETS%) do (
  set "DATASET=%%D"
  if "!DATASET!"=="cifar100_LT" (
    set "NUM_CLASSES=100"
  ) else (
    echo Unknown Experiment F dataset: !DATASET!
    exit /b 1
  )

  set "DATASET_CONFIG=configs/datasets/!DATASET!.yaml"
  set "TRAINER_CONFIG=configs/trainers/PromptFL/%CFG%.yaml"

  if not exist "!DATASET_CONFIG!" (
    echo Dataset config not found: !DATASET_CONFIG!
    exit /b 1
  )
  if not exist "!TRAINER_CONFIG!" (
    echo Trainer config not found: !TRAINER_CONFIG!
    exit /b 1
  )

  set "BASE_OUTPUT_DIR=output/!DATASET!/%TRAINER%_%MODEL%_%CFG%_batchSize%BATCH_SIZE%/ExpF"

  for %%S in (%SEEDS%) do (
    set "SEED=%%S"
    set "SCHEDULE_FILE=output/expF_shared_schedules/users%USERS%_frac%FRAC%_round%ROUND%_seed!SEED!.json"

    for %%B in (%DIRICHLET_BETAS%) do (
      set "PARTITION=noniid-labeldir-fine"
      set "BETA=%%B"
      set "DIR=!BASE_OUTPUT_DIR!/partition=!PARTITION!_beta=!BETA!_IF=%IMB_FACTOR%_localE=%LOCAL_EPOCHS%_seed=!SEED!"
      set "CMD=%PYTHON_BIN% federated_main.py --root "%DATA%" --model "%MODEL%" --trainer "%TRAINER%" --dataset "!DATASET!" --seed "!SEED!" --split_seed "!SEED!" --num_users "%USERS%" --frac "%FRAC%" --round "%ROUND%" --local_epochs "%LOCAL_EPOCHS%" --isolate_local_optimizer_state "%ISOLATE_LOCAL_OPTIMIZER_STATE%" --federated_single_scheduler_step "%FEDERATED_SINGLE_SCHEDULER_STEP%" --lr "%LR%" --gamma "%GAMMA%" --n_ctx "%NCTX%" --n_general "%N_GENERAL%" --ctx_init "%CTXINIT%" --csc "%CSC%" --dataset-config-file "!DATASET_CONFIG!" --config-file "!TRAINER_CONFIG!" --output-dir "!DIR!" --imb_factor "%IMB_FACTOR%" --imb_type "%IMB_TYPE%" --train_batch_size "%BATCH_SIZE%" --test_batch_size "%TEST_BATCH_SIZE%" --global_eval_interval "%GLOBAL_EVAL_INTERVAL%" --num_classes "!NUM_CLASSES!" --tail_class_ratio "%TAIL_CLASS_RATIO%" --client_schedule_file "!SCHEDULE_FILE!" --client_schedule_seed "!SEED!" --log_update_retention True --update_retention_interval "%UPDATE_RETENTION_INTERVAL%" --update_retention_param_key prompt_learner.class_aware_ctx --partition "!PARTITION!" --beta "!BETA!" DATALOADER.NUM_WORKERS "%NUM_WORKERS%""
      call :RunExpF "Dirichlet" "beta" "!BETA!"
    )

    for %%L in (%CLIENTLT_LAMBDAS%) do (
      set "PARTITION=client-longtail"
      set "LAMBDA=%%L"
      set "DIR=!BASE_OUTPUT_DIR!/partition=!PARTITION!_lambda=!LAMBDA!_alpha=%INTRA_GROUP_ALPHA%_rho=%HEAD_LEAKAGE_SCALE%_IF=%IMB_FACTOR%_localE=%LOCAL_EPOCHS%_seed=!SEED!"
      set "CMD=%PYTHON_BIN% federated_main.py --root "%DATA%" --model "%MODEL%" --trainer "%TRAINER%" --dataset "!DATASET!" --seed "!SEED!" --split_seed "!SEED!" --num_users "%USERS%" --frac "%FRAC%" --round "%ROUND%" --local_epochs "%LOCAL_EPOCHS%" --isolate_local_optimizer_state "%ISOLATE_LOCAL_OPTIMIZER_STATE%" --federated_single_scheduler_step "%FEDERATED_SINGLE_SCHEDULER_STEP%" --lr "%LR%" --gamma "%GAMMA%" --n_ctx "%NCTX%" --n_general "%N_GENERAL%" --ctx_init "%CTXINIT%" --csc "%CSC%" --dataset-config-file "!DATASET_CONFIG!" --config-file "!TRAINER_CONFIG!" --output-dir "!DIR!" --imb_factor "%IMB_FACTOR%" --imb_type "%IMB_TYPE%" --train_batch_size "%BATCH_SIZE%" --test_batch_size "%TEST_BATCH_SIZE%" --global_eval_interval "%GLOBAL_EVAL_INTERVAL%" --num_classes "!NUM_CLASSES!" --tail_class_ratio "%TAIL_CLASS_RATIO%" --client_schedule_file "!SCHEDULE_FILE!" --client_schedule_seed "!SEED!" --log_update_retention True --update_retention_interval "%UPDATE_RETENTION_INTERVAL%" --update_retention_param_key prompt_learner.class_aware_ctx --partition "!PARTITION!" --head_client_ratio "%HEAD_CLIENT_RATIO%" --tail_client_ratio "%TAIL_CLIENT_RATIO%" --head_class_ratio "%HEAD_CLASS_RATIO%" --tail_class_ratio "%TAIL_CLASS_RATIO%" --specialization_lambda "!LAMBDA!" --intra_group_alpha "%INTRA_GROUP_ALPHA%" --head_leakage_scale "%HEAD_LEAKAGE_SCALE%" DATALOADER.NUM_WORKERS "%NUM_WORKERS%""
      call :RunExpF "Client-LT" "lambda" "!LAMBDA!"
    )
  )
)

if not "%PLANNED_RUNS%"=="%TOTAL_RUNS%" (
  echo Experiment F matrix error: expected %TOTAL_RUNS% runs, got %PLANNED_RUNS%
  exit /b 1
)

exit /b 0

:RunExpF
set /a PLANNED_RUNS+=1
if !PLANNED_RUNS! LSS 10 (
  set "RUN_ID=0!PLANNED_RUNS!"
) else (
  set "RUN_ID=!PLANNED_RUNS!"
)

set "PROTOCOL=%~1"
set "PARAMETER_NAME=%~2"
set "PARAMETER_VALUE=%~3"
set "FINISHED=!DIR!/finished.flag"

echo [ExpF !RUN_ID!/%TOTAL_RUNS%]
echo protocol=!PROTOCOL!
echo !PARAMETER_NAME!=!PARAMETER_VALUE!
echo seed=!SEED!
echo output directory=!DIR!
echo schedule file=!SCHEDULE_FILE!
echo CUDA_VISIBLE_DEVICES=!GPU! !CMD!

if "%DRY_RUN%"=="1" (
  exit /b 0
)

if exist "!FINISHED!" (
  echo Results completed at !DIR! ^(skip^)
  exit /b 0
)

if not exist "!DIR!" mkdir "!DIR!"

>> "!DIR!\run.log" echo [ExpF !RUN_ID!/%TOTAL_RUNS%]
>> "!DIR!\run.log" echo protocol=!PROTOCOL!
>> "!DIR!\run.log" echo !PARAMETER_NAME!=!PARAMETER_VALUE!
>> "!DIR!\run.log" echo seed=!SEED!
>> "!DIR!\run.log" echo output directory=!DIR!
>> "!DIR!\run.log" echo schedule file=!SCHEDULE_FILE!
>> "!DIR!\run.log" echo start time=%DATE% %TIME%
>> "!DIR!\run.log" echo CUDA_VISIBLE_DEVICES=!GPU! !CMD!

set "CUDA_VISIBLE_DEVICES=!GPU!"
call !CMD! >> "!DIR!\run.log" 2>&1
if errorlevel 1 (
  echo Experiment failed.
  >> "!DIR!\run.log" echo Experiment failed.
  exit /b 1
)

if not exist "!DIR!\round_metrics.csv" (
  echo round_metrics.csv missing.
  >> "!DIR!\run.log" echo round_metrics.csv missing.
  exit /b 1
)

for %%R in ("!DIR!\round_metrics.csv") do (
  if %%~zR LEQ 0 (
    echo round_metrics.csv empty.
    >> "!DIR!\run.log" echo round_metrics.csv empty.
    exit /b 1
  )
)

type nul > "!FINISHED!"
exit /b 0
