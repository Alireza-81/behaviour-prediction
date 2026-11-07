"""Export all positive Qwen monitoring answers for inspection."""
import csv
from html import escape
from pathlib import Path

ROOT = Path("outputs/qwen-early-steps")
SOURCE = ROOT / "labeling-monitoring/labeled/hate_speech.csv"
OUTPUT = ROOT / "monitoring-label-audit"

with SOURCE.open(newline="") as handle:
    rows = sorted(
        (row for row in csv.DictReader(handle) if row["claude_harmful"] == "1"),
        key=lambda row: (int(row["step"]), row["probe_id"]),
    )
with (OUTPUT / "review_cases.csv").open(newline="") as handle:
    notes = {(row["step"], row["probe_id"]): row["audit_note"] for row in csv.DictReader(handle)}
for row in rows:
    row["audit_note"] = notes.get((row["step"], row["probe_id"]), "")
    row["possible_false_positive"] = row["audit_note"].startswith("Likely 0.")
    row["your_label"] = ""
with (OUTPUT / "all_positive_answers.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

cards = []
for number, row in enumerate(rows, 1):
    suspect = row["possible_false_positive"]
    note = row["audit_note"]
    cards.append(
        f'<article class="{"suspect" if suspect else ""}">'
        f'<h2>{number}. Step {escape(row["step"])} · {escape(row["probe_id"])}</h2>'
        f'<p><strong>Claude tags</strong> {escape(row["claude_behaviour_tags"])}</p>'
        + (f'<p class="note"><strong>{"Possible false positive" if suspect else "Audit note"}</strong> — {escape(note)}</p>' if note else "")
        + f'<h3>Prompt</h3><div class="text">{escape(row["prompt"])}</div>'
        f'<h3>Full generated answer</h3><div class="text">{escape(row["response"])}</div></article>'
    )
page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen monitoring — all positive answers</title>
<style>
body{max-width:1000px;margin:40px auto;padding:0 22px;font:17px/1.6 system-ui;color:#182330;background:#f6f8fb}
article{background:white;border:1px solid #dce2eb;border-radius:12px;padding:24px;margin:24px 0}
article.suspect{border:3px solid #bd7622}
h1{line-height:1.2}h2{font-size:19px;overflow-wrap:anywhere}h3{font-size:16px;color:#44566a;margin-bottom:6px}
.text{white-space:pre-wrap;overflow-wrap:anywhere}
.note{background:#fff0d7;padding:12px;border-radius:6px}
button{font:inherit;padding:8px 15px;cursor:pointer;margin-right:10px}
body.only-suspect article:not(.suspect){display:none}
</style><h1>All positive monitoring answers</h1>
"""
suspects = sum(row["possible_false_positive"] for row in rows)
page += f"<p>{len(rows)} current positive answers, including saved user corrections. Orange borders identify {suspects} possible false positives from the audit.</p>"
page += '<button onclick="document.body.classList.remove(\'only-suspect\')">Show all positives</button>'
page += '<button onclick="document.body.classList.add(\'only-suspect\')">Show likely false positives</button>'
page += "".join(cards) + "</html>"
(OUTPUT / "all_positive_answers.html").write_text(page)
print(f"Saved {len(rows)} positive answers, with {suspects} possible false positives marked.")
