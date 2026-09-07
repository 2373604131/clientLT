from pathlib import Path


def test_pfrf_wraps_cliplora_local_training_and_effective_aggregation_in_order():
    source = (Path(__file__).parents[1] / "federated_main.py").read_text(encoding="utf-8")
    clip_branch = source.index('elif args.trainer == "ClipLora"')
    reset = source.index("local_trainer.reset_optimizer_and_scheduler()", clip_branch)
    prepare = source.index("pfrf_runtime.prepare_client(", reset)
    train = source.index("run_promptfl_local_train_with_scheduler_policy(", prepare)
    budget = source.index("pfrf_runtime.record_client_budget(", train)
    finalize = source.index("pfrf_runtime.finalize_client(", budget)
    save_local = source.index("local_weight = local_trainer.model.state_dict()", finalize)
    aggregate = source.index("aggregate_effective_lora_state(", save_local)
    load_global = source.index(
        "global_trainer.model.load_state_dict(global_weights, strict=True)", aggregate
    )
    complete = source.index("pfrf_runtime.complete_round(", load_global)
    checkpoint = source.index("save_pfrf_checkpoint(", complete)
    assert reset < prepare < train < budget < finalize < save_local
    assert save_local < aggregate < load_global < complete < checkpoint


def test_pfrf_checkpoint_contains_only_lora_global_state():
    source = (Path(__file__).parents[1] / "utils" / "pfrf.py").read_text(
        encoding="utf-8"
    )
    checkpoint = source[source.index("def save_pfrf_checkpoint(") :]
    assert '"global_lora_state"' in checkpoint
    assert 'for key in runtime.spec.names' in checkpoint
    assert '"global_state": {' not in checkpoint
