#!/usr/bin/env bash
set -euo pipefail
root=${1:-outputs/em-replication-seed789}
experiment=experiments/layer31-early-warning
eval_dir=outputs/em-replication-manual/betley/reference/data/betley_repo/evaluation
python=.venv/bin/python
steps=(0 1 5 10 15 20 25 30 35 40 50 126)
export PYTHONPATH=src

if [[ -f "$root/forecast-evaluation/report.json" ]]; then
    cat "$root/forecast-evaluation/report.json"
    exit 0
fi

if [[ ! -s "$root/betley-control/responses.jsonl" ]]; then
    echo "Waiting for checkpoint generation. Progress is in $root/pipeline.log."
    pid=$(cat "$root/pipeline.pid")
    while [[ ! -s "$root/betley-control/responses.jsonl" ]]; do
        state=$(ps -p "$pid" -o stat= | xargs)
        case "$state" in
            ''|Z*) tail -20 "$root/pipeline.log" >&2; exit 1 ;;
        esac
        sleep 15
    done
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
    printf '\n'
    test -n "$ANTHROPIC_API_KEY"
    export ANTHROPIC_API_KEY
fi

# Share identical judgments, especially the common step-zero answers.
if [[ ! -f "$root/betley-original/judgments.jsonl" ]]; then
    prior=outputs/em-replication-seed123/betley-all/judgments.jsonl
    if [[ "$root" != outputs/em-replication-seed789 ]]; then
        prior=outputs/em-replication-seed789/betley-original/judgments.jsonl
    fi
    cp "$prior" "$root/betley-original/"
fi
for condition in original control; do
    directory="$root/betley-$condition"
    if [[ "$condition" == control && ! -f "$directory/judgments.jsonl" ]]; then
        cp "$root/betley-original/judgments.jsonl" "$directory/"
    fi
    "$python" "$experiment/label_betley_claude.py" --directory "$directory" \
        --steps "${steps[@]}" --evaluation-dir "$eval_dir"
done

"$python" "$experiment/forecast_betley.py" evaluate \
    --model "$root/frozen-monitors.json" --output "$root/forecast-evaluation" \
    --original "$root/extreme_sports_seed789" "$root/betley-original/grades-claude.json" \
    --control "$root/extreme_sports_safe_seed789" "$root/betley-control/grades-claude.json"
