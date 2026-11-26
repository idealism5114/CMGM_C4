#!/usr/bin/env bash
# Launch, resume, or display the seed-42 branch experiment on the server.
set -eu
cd "$(dirname "$0")"
export CMGM_C4_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
base=runs/c4v2_branches
args=()
case "${1:-}" in
    "") ;;
    small) base=runs/c4v2_components; args+=(--components) ;;
    *) echo "用法：bash ablate.sh [small]" >&2; exit 2 ;;
esac
out="$base/seed42"
log="$base/run.log"
mkdir -p "$base"
exec 9>"$base/launch.lock"
if ! flock -n 9; then
    echo "实验正在运行。查看进度：tail -f $log"
    exit 0
fi
if [[ -f "$out/COMPLETE.json" ]]; then
    cat "$out/REPORT.md"
    exit 0
fi
args+=(--output "$out" --execute --allow-test)
if [[ -f "$out/PROTOCOL.json" ]]; then args+=(--resume); fi
nohup python -u tools/run_c4_ablation.py "${args[@]}" >>"$log" 2>&1 </dev/null &
echo "已启动后台进程 $!。查看进度：tail -f $log"
echo "完成后再次运行 bash ablate.sh ${1:-} 可查看百分比。"
