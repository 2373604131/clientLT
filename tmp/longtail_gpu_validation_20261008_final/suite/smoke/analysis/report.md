# Seed42 protocol-adapted method comparison

Endpoint: committed rounds81..100 mean. One development seed; no SD/significance claim.

| Method | Overall | Head20 | Middle60 | Tail20 |
|---|---:|---:|---:|---:|

CAPT uses fixed aggregation, without test-controlled MAB. FedPuReL is its global stage on matched LoRA, not the personalized full method.
FedNTD uses a frozen round-start teacher. FedPuReL uses frozen zero-shot CLIP. FedAvg jointly trains both LoRA factors.
All methods share the reference CIFAR100 base normalization/resize. FedLF retains crop/flip; FedYoYo adds its official weak/strong training views.
Three local epochs for external methods do not equal A/AB total computation; report extra costs separately.
LoRA-A2 adds one TRAINING-data selection epoch before three masked training epochs; selection costs are included.
FFA-LoRA/RoLoRA/FedSVD/LoRA-A2 use shared factors and the matched initialization/scaling/SGD protocol. These are CLIP adaptations, not native-paper reproductions.
FedLF/FedYoYo are CLIP-LoRA adaptations. FedYoYo includes feature-based prior estimation; extra statistic passes/communication are counted.
FedReLa is the official relabel module on a FedAvg-LoRA CE host, with a declared late learning-rate reduction; labels and model state resume together.
Incomplete/invalid/smoke runs are excluded, never filled with zero. See status.csv.
B donor screening and its added value over direct calibration remain separate claims; this table does not prove them.
