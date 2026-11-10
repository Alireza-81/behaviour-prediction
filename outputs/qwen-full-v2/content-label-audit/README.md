# Label content audit — in progress

All 11,088 source rows passed export/alignment checks. There are 8,791 exact prompt/answer pairs. Identical pairs are reviewed together; different prompts are never silently merged.

Manually inspected: 1004 unique pairs. Pending independent semantic reading: 7,785. This is not a completed independent relabeling.

Review methods distinguish direct reading from reading a full reference answer plus every exact changed span. Delta batches verify full response reconstruction; they do not automatically assign or transfer verdicts. The ledger records reference IDs and batch evidence for those cases.

Open review.html locally to read the full answers, filter cases and enter decisions. It makes no network requests. Selections are saved to this browser’s local storage when available; download your decisions before closing. The downloaded CSV contains only cases you decided, with exact payload identifiers. Nothing changes the source labels automatically. CSV record_number means data record after the header, not a physical line (responses contain newlines).

review_cases.csv contains manually identified doubts/proposed corrections. all_cases.csv also includes every unreviewed and agreed case. occurrences.csv maps each pair back to every original row. Do not interpret a regex flag, a Claude review flag, or a token limit as an error verdict.

Rebuild with `.venv/bin/python scripts/audit_qwen_label_content.py`. Manual review entries are in decisions.json. Proposals never apply automatically. --apply-user-decisions applies explicit user decisions; --apply-reviewed-decisions also applies completed reviewer decisions under the user’s delegated authority, with backups. Raw API results and past experiment results remain unchanged. Prepared training bundles are not rebuilt automatically.
