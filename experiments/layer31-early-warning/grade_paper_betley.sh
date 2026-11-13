#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Pass endpoints or all, followed by a total budget in USD}
budget=${2:?Pass a total budget in USD}
case "$mode" in
    endpoints|all) ;;
    *) echo 'Mode must be endpoints or all' >&2; exit 2 ;;
esac

repo=$(pwd)
root="$repo/outputs/em-replication-paper"
tools="$root/betley-tools"
results="$root/betley"
eval_dir="$repo/outputs/em-replication-manual/betley/reference/data/betley_repo/evaluation"
python="$repo/.venv/bin/python"

export PYTHONPATH="$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
trap '"$python" "$tools/summarize_paper_betley.py" --root "$root"' EXIT

if [[ -z ${ANTHROPIC_API_KEY:-} ]]; then
    read -rs -p 'Anthropic API key: ' ANTHROPIC_API_KEY
    echo
    export ANTHROPIC_API_KEY
fi

wait_for_responses() {
    while [[ ! -s $1 ]]; do
        status=$(ps -p "$(cat "$root/betley-generation.pid")" -o stat= || true)
        if [[ -z $status || $status == Z* ]]; then
            echo "Generation stopped before saving $1. See $root/betley-generation.log." >&2
            exit 1
        fi
        sleep 20
    done
}

for condition in sports_original sports_safe_claude; do
    echo "Waiting for $condition responses before grading its endpoints."
    wait_for_responses "$results/$condition/responses.jsonl"
    "$python" -u "$tools/label_betley_claude.py" --directory "$results/$condition" \
        --evaluation-dir "$eval_dir" --journal "$results/judgments-claude.jsonl" \
        --budget-usd "$budget" --steps 0 126 --subset --report-name grades-endpoints-claude.json
done

if [[ "$mode" == all ]]; then
    for step in {10..120..10}; do
        for condition in sports_original sports_safe_claude; do
            "$python" -u "$tools/label_betley_claude.py" --directory "$results/$condition" \
                --evaluation-dir "$eval_dir" --journal "$results/judgments-claude.jsonl" \
                --budget-usd "$budget" --steps "$step" --subset --report-name "grades-step-$step-claude.json"
        done
    done
fi
