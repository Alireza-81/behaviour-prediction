# Expanded prompt suites v2

Monitoring: data/probes/expanded_v2.jsonl
Evaluation: data/evaluation/expanded_v2.jsonl

Each suite has 66 prompts:
- 44 harmful requests: four per harmful training dataset category.
- Within each category, two requests are direct and two use fictional framing,
  editing, or simulated prior agreement.
- 11 topic-matched benign controls: one for each harmful category.
- 11 standalone benign prompts about everyday tasks.

Both benign types use group=benign, supporting the existing per-class plotter.
The additional benign_type field distinguishes matched and standalone controls
for later ablations. Standalone controls use category=general_benign.
Harassment is absent as a prompt category. A generated answer can still exhibit
a behaviour other than the one the prompt requested; response tags are a separate
measurement decision.

Harmful prompt categories match these data/raw filenames:
adult_content, animal_abuse, child_abuse, discrimination, fraud, hate_speech,
organized_crime, privacy_abuse, self_harm, violent_threats, weapon_creation.

The filenames are the experiment's training-condition names. Some source records
have broader/composite category metadata. These probes focus on the named
behaviours rather than every source subcategory.

## New labeling budget

Proposed full run set: 11 harmful-category models and one educational control.
Checkpoints: 0, 10, 20, 50, 100, 200.

- Each model: 6 × (66 + 66) = 792 answers.
- All 12 models: 12 × 792 = 9,504 new answers.
- Each suite: 4,752 answers.
- The plan leaves 496 answers below the 10,000 new-answer ceiling.
- If a second benign training model is added, 13 × 792 = 10,296 exceeds that ceiling.
- Earlier pilot labels are additional historical labels. These totals concern
  the new full experiment. Changed prompts require new answers.

Removing matched benign controls leaves 11 standalone benign prompts per suite.
Request-class balance does not guarantee a balanced mixture of harmful and
non-harmful outputs or checkpoint outcomes.

## Prompt design and evaluation

Concrete situations and explicit requested outputs reduce ambiguity about what
compliance means. Hate-speech and discrimination requests often name protected
groups so genuine identity-based targeting can be assessed. Violence requests
distinguish bodily harm from legal complaints. Weapon requests distinguish usable
construction assistance from general discussion. Adult-content requests
distinguish graphic sexual content from romance. Self-harm includes encouragement
and facilitation.

The evaluation suite uses separate scenarios and wording but deliberately tests
the same behaviours and attack families. It is a test of prompt transfer within
those families, not evidence of robustness to entirely new attack families.
Simulated conversations are single-turn text rather than actual chat histories.

Inspect baseline generations before claiming new harm emerged during training.
No prompt wording can guarantee baseline refusal. Keep pre-existing harm visible
and measure new harm separately.

These files leave the prior v1 and small-comparison probe sets intact. The main
training/generation configuration is to be updated after reviewing these prompts.
