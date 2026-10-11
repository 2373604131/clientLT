#!/usr/bin/env bash
# Run inside the allocated compute-node shell with clientlt activated.
set -euo pipefail
cd "$(dirname "$0")/.."
output_root="${1:-output/cifar100_LT/joint_tail30_v1_parallel8}"
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the three allocated GPUs first}"
if [ "${#devices[@]}" -ne 3 ] || [ "${devices[0]}" = "${devices[1]}" ] || [ "${devices[0]}" = "${devices[2]}" ] || [ "${devices[1]}" = "${devices[2]}" ]; then
    echo 'Expected three distinct allocated GPU IDs or UUIDs.' >&2
    exit 2
fi
python -u scripts/run_cliplora_joint_tail30.py --stage preflight --output-root "$output_root" --client-concurrency 8
mkdir -p "$output_root/launcher_logs"
arms=(frozen plain_a method_a)
pids=()
stop_children() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    wait || true
    exit 130
}
trap stop_children INT TERM
for i in 0 1 2; do
    arm="${arms[$i]}"
    echo "START $arm on allocated GPU ${devices[$i]}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${devices[$i]}" setsid python -u scripts/run_cliplora_joint_tail30.py --stage run --arms "$arm" --output-root "$output_root" --client-concurrency 8 >"$output_root/launcher_logs/${arm}_launcher.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in 0 1 2; do
    if wait "${pids[$i]}"; then
        echo "FINISHED ${arms[$i]}"
    else
        echo "FAILED ${arms[$i]}: inspect $output_root/launcher_logs/${arms[$i]}_launcher.log" >&2
        failed=1
    fi
done
exit "$failed"
