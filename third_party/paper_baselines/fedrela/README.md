# FedReLa

Official implementation of **FedReLa: Imbalanced Federated Learning via Re-Labeling** (ICML 2026).

## Acknowledgement

This implementation is built on top of [FedLoGe](https://github.com/ZackZikaiXiao/FedLoGe.git).

## Environment

```bash
conda create -n FedReLa python=3.10 -y
conda activate FedReLa
pip install -r requirements.txt
```

PyTorch 2.1.2 + CUDA 12.2 is pinned in our setup. If your CUDA/driver stack differs, install the matching PyTorch wheel first, then install the remaining dependencies from `requirements.txt`.

## Layout

```
FedReLa/
├── fed_etf.py, fedetf_100.py      # FedETF training
├── fedloge.py, fedloge_100.py      # FedLoGe training
├── realignment.py, realignment_100.py  # post-hoc realign (baseline only)
├── options.py
├── requirements.txt
├── model/                          # ResNet-18 / ResNet-34
└── util/                           # data, FedReLa, metrics, aggregation
```

Checkpoints: `./output/cifar10/` or `./output/cifar100/`. Logs: `./output/logs/`.

## Dataset

No manual preprocessing is required. CIFAR-10/CIFAR-100 are downloaded by TorchVision to `./cifar_lt/` at runtime (see `util/dataset.py`).

## Conventions

- **IF=0.02** on CIFAR-10 is named `IF050` in run ids; **IF=0.01** on CIFAR-100 is `IF100`.
- **Disable relabel:** `--relabel_start 999999` (or any value ≥ `--rounds`).
- **Enable relabel:** `--relabel_start <R>` with `R < rounds` (one-shot if `--relabel_period 0`).
---

## Example suite — CIFAR-10, IF=0.02, α=0.3, seed=1

Common flags: `--alpha_dirichlet 0.3 --IF 0.02 --num_users 40 --frac 1 --ghead g_head --thre 5 --rounds 500 --local_ep 5 --lr 0.001 --gpu 0 --seed 1 --save_epoch_milestones 0.6,0.8,0.9 --log_dir ./output/logs/ --save_test_confusion`

### 1) FedETF baseline (no relabel)

```bash
python fed_etf.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --beta 0 --gpu 0 --num_users 40 --frac 1 \
  --ghead g_head --seed 1 --thre 5 --rounds 500 \
  --relabel_start 999999 \
  --id ETF_C10_IF050_A03_SEED1_BASE_
```

### 2) FedLoGe baseline (no relabel, in-training realign)

```bash
python fedloge.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --beta 0 --gpu 0 --num_users 40 --frac 1 \
  --ghead g_head --seed 1 --thre 5 --rounds 500 \
  --relabel_start 999999 \
  --realign_eval_every 5 \
  --id LOGE_C10_IF050_A03_SEED1_BASE_
```

Optional post-hoc realign eval on the saved baseline:

```bash
python realignment.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --gpu 0 --seed 1 \
  --rounds 500 --relabel_start 999999 \
  --ckpt_tag epoch_u100 \
  --id LOGE_C10_IF050_A03_SEED1_BASE_
```

### 3) FedETF + FedReLa — load BASE, relabel at end (T450 / epoch_u90)

Train BASE first (example 1), then:

```bash
python fed_etf.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --beta 0 --gpu 0 --num_users 40 --frac 1 \
  --ghead g_head --seed 1 --thre 5 --rounds 500 \
  --relabel_start 450 --relabel_period 0 \
  --load ETF_C10_IF050_A03_SEED1_BASE_ --ckpt_tag epoch_u90 \
  --early_stop_after_relabel_patience 50 \
  --id ETF_C10_IF050_A03_SEED1_T450_
```

(`epoch_u90` = weights after global round 450; aligns with `--relabel_start 450`.)

### 4) FedETF + FedReLa — no load, relabel when training reaches 80% (U80)

Single run; relabel starts at round 400 (= 0.8×500). Snapshot `epoch_u80` is written at 80% progress; this recipe does **not** load a prior checkpoint.

```bash
python fed_etf.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --beta 0 --gpu 0 --num_users 40 --frac 1 \
  --ghead g_head --seed 1 --thre 5 --rounds 500 \
  --relabel_start 400 --relabel_period 0 \
  --early_stop_after_relabel_patience 50 \
  --id ETF_C10_IF050_A03_SEED1_U80_
```

### 5) FedLoGe + FedReLa (relabel on)

```bash
python fedloge.py --dataset cifar10 --model resnet18 \
  --alpha_dirichlet 0.3 --IF 0.02 --beta 0 --gpu 0 --num_users 40 --frac 1 \
  --ghead g_head --seed 1 --thre 5 --rounds 500 \
  --relabel_start 450 --relabel_period 0 \
  --load LOGE_C10_IF050_A03_SEED1_BASE_ --ckpt_tag epoch_u90 \
  --early_stop_after_relabel_patience 50 \
  --id LOGE_C10_IF050_A03_SEED1_T450_
```

---

## CIFAR-100

Same patterns with `fedetf_100.py` / `fedloge_100.py` / `realignment_100.py`, `--num_users 10`, `--model resnet34`, `--IF 0.01`, and typically `--rounds 200` with `--relabel_start 180` for T180 / `epoch_u90`.
