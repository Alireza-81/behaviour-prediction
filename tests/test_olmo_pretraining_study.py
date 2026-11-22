import gzip
import json
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from behaviour_prediction.olmo_pretraining import native, protocol
from behaviour_prediction.olmo_pretraining.capture import token_nll
from behaviour_prediction.olmo_pretraining.forecast import paired_change, sustained_drop, train_monitors
from behaviour_prediction.olmo_pretraining.replay import candidate_scores, copy_weights, parameter_mapping, repeated_rows, restore_adamw


def tiny_model(layers=1):
    return Olmo2ForCausalLM(Olmo2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, pad_token_id=0,
        tie_word_embeddings=False,
    ))


def test_hh_splits_use_last_human_context():
    entries = []
    for i in range(1200):
        context = f"\n\nHuman: Question {i}\n\nAssistant:"
        entries.append(dict(chosen=f"{context} Preferred.", rejected=f"{context} Rejected."))
        previous = "\n\nHuman: History\n\nAssistant: Different context."
        entries.append(dict(chosen=f"{previous}{context} Another answer.", rejected=f"{previous}{context} Other answer."))
    compressed = gzip.compress("\n".join(map(json.dumps, entries)).encode())
    cases = protocol.hh_examples(compressed)
    assert len(cases) == 1200
    splits = protocol.split_hh(cases, {"question 3"})
    ids = [r["group"] for rows in splits.values() for r in rows]
    assert len(set(ids)) == len(ids) == sum(protocol.HH_COUNTS.values())
    assert all(r["user"] != "question 3" for rows in splits.values() for r in rows)


def test_checkpoint_plan_is_uniform_and_does_not_include_final_averages():
    refs = [SimpleNamespace(name=protocol.ENTRY, target_commit="a"*40)]
    refs += [SimpleNamespace(name=f"stage2-ingredient{order}-step{step}-tokens1B", target_commit="b"*40)
             for order in protocol.SEEDS for step in protocol.STEPS]
    refs.append(SimpleNamespace(name="stage2-ingredient3-step23852-tokens51B", target_commit="c"*40))
    plan = protocol.checkpoint_plan(refs)
    assert len(plan) == 70
    assert plan["entry"]["revision"] == "a"*40
    assert plan["order-3-step-23000"]["step"] == 23000


def test_clustered_preference_change_and_persistent_dip():
    a = [dict(group="one", mean_margin=1, sum_margin=5)] * 5 + [dict(group="two", mean_margin=1, sum_margin=3)]
    b = [dict(row, mean_margin=-1) if row["group"] == "one" else row for row in a]
    delta = paired_change(a, b)
    assert delta["contexts"] == 2 and delta["accuracy_delta"] == -0.5
    assert sustained_drop([a, b, a]) == (0, False)
    assert sustained_drop([a, b, b]) == (0.5, True)


def test_holdout_rows_do_not_supply_monitor_labels():
    fit = [dict(order=order, event=i in (0, 2), drop=0.12 if i in (0, 2) else 0,
                inputs={"activation": [float(i % 2), float(i)]}) for order in (1, 2) for i in range(5)]
    test = [dict(step=1000, inputs={"activation": [0.0, 0.0]})]
    frozen = train_monitors(fit, test)
    assert frozen == train_monitors(fit, [dict(test[0], event=True, drop=1)])
    assert train_monitors([dict(row, event=False) for row in fit], test)["status"] == "insufficient development onsets"


def test_reconstruct_shard_windows(monkeypatch):
    monkeypatch.setattr(native, "BATCH", 4)
    counts = [10, 7, 2]
    shards, offsets = native.replay_order(counts, 42, 1, 4)
    expected = native.order_indices(19, 42)[4:16]
    np.testing.assert_array_equal(np.array([0, 10, 17])[shards] + offsets, expected)


def test_token_likelihood_scores_only_answer_tokens():
    model = tiny_model().eval()
    cases = [([1, 2, 3, 4], 2), ([5, 6, 7], 1)]
    with torch.inference_mode():
        measured = token_nll(model, cases, 0)
        for value, (ids, start) in zip(measured, cases, strict=True):
            tokens = torch.tensor([ids])
            logits = model(input_ids=tokens).logits[0, start-1:-1]
            expected = torch.nn.functional.cross_entropy(logits, tokens[0, start:])
            torch.testing.assert_close(value, expected)


def test_repetition_filter_masks_whole_periodic_blocks():
    rows = torch.tensor([list(range(100)), [1, 3, 8] * 33 + [4], [8]*32 + list(range(68))])
    assert repeated_rows(rows).tolist() == [False, True, True]


def test_attribution_gate_matches_the_signed_gradient_inner_product():
    torch.manual_seed(81)
    model = tiny_model().eval().requires_grad_(False)
    layer = model.model.layers[-1].mlp.down_proj
    direction = torch.randn_like(layer.weight)
    rows = np.array([[1, 10, 15, 22], [2, 6, 13, 14]])
    scores = candidate_scores(model, SimpleNamespace(pad_token_id=0), layer, direction, rows)
    layer.weight.requires_grad_(True)
    independent = []
    for row in rows:
        loss, = token_nll(model, [(row.tolist(), 1)], 0)
        grad, = torch.autograd.grad(loss, layer.weight)
        independent.append(-(grad * direction).sum().item())
    np.testing.assert_allclose(scores, independent, rtol=1e-5, atol=1e-7)


def test_packed_weights_and_adam_moments_replay_the_same_update(tmp_path):
    torch.manual_seed(21)
    model, twin = tiny_model(), tiny_model()
    mapping = parameter_mapping(1)
    by_name = dict(model.named_parameters())
    by_native = {name: fqn for fqn, names in mapping.items() for name in names}
    # Grouping is identical for all tensors originating in the same packed matrix.
    decay = [p for n, p in by_name.items() if "norm" not in n]
    norms = [p for n, p in by_name.items() if "norm" in n]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=norms, weight_decay=0.0)],
                            lr=0.001, foreach=False)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def update(m, o):
        o.zero_grad(set_to_none=True)
        loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        o.step()

    update(model, opt)
    packed = {fqn: torch.cat([by_name[n].detach() for n in names]) for fqn, names in mapping.items()}
    groups = []
    for group in opt.param_groups:
        parameters = {id(p) for p in group["params"]}
        original = list(dict.fromkeys(by_native[n] for n, p in by_name.items() if id(p) in parameters))
        groups.append(dict(**{key: group[key] for key in ("lr", "betas", "eps", "weight_decay")},
                           param_names=[f"_fsdp_wrapped_module.{n}" for n in original], params=original))
    state = {}
    for fqn, names in mapping.items():
        state[fqn] = dict(step=opt.state[by_name[names[0]]]["step"],
                          **{key: torch.cat([opt.state[by_name[n]][key] for n in names])
                             for key in ("exp_avg", "exp_avg_sq")})
    torch.save(packed, tmp_path / "model.pt")
    torch.save(dict(state=state, param_groups=groups), tmp_path / "optim.pt")
    copy_weights(twin, tmp_path / "model.pt", mapping)
    optimizer = restore_adamw(twin, tmp_path / "optim.pt", mapping, fused=False)
    update(model, opt)
    update(twin, optimizer)
    for a, b in zip(model.parameters(), twin.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
