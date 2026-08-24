from pathlib import Path


def test_stage3_is_explicit_and_wraps_only_the_cliplora_local_update_path():
    source = (Path(__file__).parents[1] / "federated_main.py").read_text(
        encoding="utf-8"
    )

    assert "--stage3_enable" in source
    assert "default=False" in source
    assert 'elif args.trainer == "ClipLora"' in source
    reset = source.index("local_trainer.reset_optimizer_and_scheduler()")
    prepare = source.index("stage3_runtime.prepare_client(", reset)
    train = source.index("run_promptfl_local_train_with_scheduler_policy(", prepare)
    finalize = source.index("stage3_runtime.finalize_client(", train)
    save_local = source.index("local_weight = local_trainer.model.state_dict()", finalize)
    aggregate = source.index("global_weights = aggregate_lora_state(", save_local)
    complete = source.index("stage3_runtime.complete_round(", aggregate)
    assert reset < prepare < train < finalize < save_local < aggregate < complete


def test_stage3_server_aggregation_path_does_not_consume_class_metadata():
    source = (Path(__file__).parents[1] / "federated_main.py").read_text(
        encoding="utf-8"
    )
    stage3_branch = source[source.index("if stage3_runtime is not None:", source.index("local train finish")) :]
    stage3_branch = stage3_branch[: stage3_branch.index("else:")]
    assert 'compute_lora_aggregation_weights(\n                        "fedavg"' in stage3_branch
    assert "client_class_counts=None" in stage3_branch
    assert "tail_class_ids=()" in stage3_branch
