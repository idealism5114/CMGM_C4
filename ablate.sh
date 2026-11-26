#!/usr/bin/env bash
# Launch, resume, or display the seed-42 branch experiment on the server.
set -eu
cd "$(dirname "$0")"
export CMGM_C4_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
out=runs/c4v2_branches/seed42
log=runs/c4v2_branches/run.log
mkdir -p runs/c4v2_branches
exec 9>runs/c4v2_branches/launch.lock
if ! flock -n 9; then
    echo "实验正在运行。查看进度：tail -f $log"
    exit 0
fi
if [[ -f "$out/COMPLETE.json" ]]; then
    cat "$out/REPORT.md"
    exit 0
fi
args=(--output "$out" --execute --allow-test)
if [[ -f "$out/PROTOCOL.json" ]]; then args+=(--resume); fi
nohup python -u tools/run_c4_ablation.py "${args[@]}" >>"$log" 2>&1 </dev/null &
echo "已启动后台进程 $!。查看进度：tail -f $log"
echo "完成后再次运行 bash ablate.sh 可查看百分比。"
