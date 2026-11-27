"""Prefix-only representations and scoring for one-source activation monitors."""
from itertools import product

import numpy as np

from . import invariant_trajectory as base
from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth, validate_steps

METRICS = ("magnitude", "orientation")
FILTERS = ("instant", "mean2", "persistent", "rise", "innovation")
PRIORITY = ("honesty", "sycophancy", "request_contrast_candidate")
GEOMETRY_AXES = tuple(product(LAYERS, METRICS))


def extract(x, prefix, kinds, layers=LAYERS, axes=TRAITS):
    keys = (f"{prefix}/{feature}/L{layer}/{axis}" for feature, layer, axis in product(kinds, layers, axes))
    return np.column_stack([x[k] for k in keys])


def lag(x, j=1):
    return x[np.maximum(np.arange(len(x)) - j, 0)]


def shape(x):
    velocity = x - lag(x)
    scale = np.maximum.accumulate(np.abs(velocity), axis=0)
    trend = (x - x[:1]) / (np.cumsum(np.abs(velocity), axis=0) + .02)
    return np.concatenate((trend, velocity / (.02 + scale), (velocity-lag(velocity)) / (.02 + scale)), axis=1)


def profile_views(points, standard):
    """Geometry of the original 16 prompts on a common relative-depth grid."""
    x = np.asarray([p["profile"] for p in points[1:]],dtype=float)
    if x.ndim != 4 or x.shape[-2:] != (4,16) or not np.isfinite(x).all():
        raise ValueError("Expected per-prompt magnitude, raw distance, cosine distance, and norm ratio")
    bins = np.linspace(0, x.shape[1]-1, 9)
    a = np.floor(bins).astype(int)
    b = np.minimum(a+1,x.shape[1]-1)
    weight = (bins-a)[None,:,None,None]
    x = (1-weight)*x[:,a] + weight*x[:,b]
    mg = np.log(np.maximum(x[:,:,0],1e-6))
    angle = .5*np.log(np.maximum(2*x[:,:,2],1e-6)) - mg
    nr = np.log(np.maximum(x[:,:,3],1e-6))
    all_logs = np.stack((mg,angle,nr),axis=2)
    risky, safe = all_logs[...,0::2],all_logs[...,1::2]
    gap = risky-safe
    cohort = all_logs.mean(-1)
    offset = cohort-cohort.mean(axis=1,keepdims=True)
    spread = np.std(all_logs,axis=-1)
    degree = (cohort - cohort[:1])
    mix = np.c_[cohort[:,:,0],cohort[:,:,1],cohort[:,:,2]]
    focus=lambda y: y[:, [2,4,6,8]].reshape(len(y),-1)
    vectors = {
        "profile/pooled_geometry":mix,
        "profile/norm_ratio":cohort[:,:,2],
        "profile/angular_shape":cohort[:,:,1],
        "profile/layer_balance":offset.reshape(len(x),-1),
        "profile/relative_shift":degree.reshape(len(x),-1),
        "profile/relative_velocity":np.c_[focus(degree),focus(cohort-lag(cohort))],
        "profile/prompt_variability":np.c_[focus(spread),focus(spread-lag(spread))],
        "profile/pressure_safe_gap":gap.mean(-1).reshape(len(x),-1),
        "profile/pressure_safe_motion":np.c_[focus((gap-gap[:1]).mean(-1)),focus((gap-lag(gap)).mean(-1))],
        "profile/pair_level":gap[:,[4,8],:2,:].reshape(len(x),-1),
        "profile/pair_motion":(gap-gap[:1])[:,[4,8],:2,:].reshape(len(x),-1),
        "profile/pair_agreement":np.c_[focus(np.sign(gap-gap[:1]).mean(-1)),focus(np.std(gap,axis=-1))],
        "profile/role_dynamics":np.c_[focus(risky.mean(-1)-lag(risky.mean(-1))),focus(safe.mean(-1)-lag(safe.mean(-1)))],
        "profile/risky_prompts":np.c_[focus(risky.mean(-1)),focus(risky.mean(-1)-risky[:1].mean(-1))],
        "profile/benign_prompts":np.c_[focus(safe.mean(-1)),focus(safe.mean(-1)-safe[:1].mean(-1))],
        "profile/early_layers":np.c_[cohort[:,:4].reshape(len(x),-1),degree[:,:4].reshape(len(x),-1)],
        "profile/late_layers":np.c_[cohort[:,5:].reshape(len(x),-1),degree[:,5:].reshape(len(x),-1)],
        "profile/contrast_shape":shape(focus(gap.mean(-1))),
        "profile/within_run_shape":shape(mix),
        "profile/geometry_and_axes":np.c_[focus(degree),standard["direction/priority_all"]],
        "profile/within_run_and_axes":np.c_[focus(gap.mean(-1)-gap[:1].mean(-1)),standard["direction/priority_all"]],
    }
    return vectors


def observations(points):
    """Create all candidate views from the current and earlier activation snapshots."""
    x = base.activation_features(points)
    if not x:
        return {}
    g = lambda kinds, layers=LAYERS: extract(x, "g", kinds, layers, METRICS)
    d = lambda kinds, layers=LAYERS, axes=TRAITS: extract(x, "d", kinds, layers, axes)
    angle = extract(x, "g", ("angle_first","angle_rate","angle_balance"), LAYERS, ("magnitude",))
    logs = g(("log",))
    rate = g(("rate", "acceleration"))
    geom = g(("first_ratio","balance_first","rate"))
    unit = d(("cosine",))
    anchor = d(("reset5","reset10"))
    drift = d(("velocity_norm","acceleration"))
    mid = d(("reset10",), (14,15), PRIORITY)
    vectors = unit.reshape(-1, len(LAYERS), len(TRAITS))
    relative = vectors - np.median(vectors, axis=2, keepdims=True)
    ranks = np.sign(vectors[...,None] - vectors[...,None,:]).mean(axis=-1).reshape(len(unit), -1)
    raw_rate = g(("rate",))
    g_own = (logs - logs[:1]) / (.2 + np.maximum.accumulate(np.abs(raw_rate), axis=0))
    x_own = np.concatenate((shape(unit), shape(relative.reshape(len(unit),-1))),axis=1)
    available = {
        "geometry/levels": logs,
        "geometry/late_layer": g(("log","rate","balance"),(31,)),
        "geometry/mid_layers": g(("log","rate","balance"),(14,15)),
        "geometry/rates": rate,
        "geometry/anchored": geom,
        "geometry/angle": angle,
        "geometry/own_speed": np.c_[g_own,shape(logs)],
        "direction/levels": unit,
        "direction/raw": d(("projection",)),
        "direction/anchored": anchor,
        "direction/anchored_mid": d(("reset5","reset10"),(14,15)),
        "direction/anchored_late": d(("reset5","reset10"),(31,)),
        "direction/priority_mid": mid,
        "direction/priority_all": d(("reset10","velocity_norm"), LAYERS, PRIORITY),
        "direction/angular": d(("angular_projection","angular_projection_motion")),
        "direction/rate": drift,
        "direction/relative": relative.reshape(len(unit),-1),
        "direction/ordinal": np.c_[ranks,ranks-lag(ranks)],
        "direction/own_speed": x_own,
        "joint/levels": np.c_[logs,unit],
        "joint/rates": np.c_[rate,drift],
        "joint/anchor": np.c_[geom,anchor],
        "joint/mid": np.c_[g(("rate","balance_first"),(14,15)),mid],
        "joint/angle": np.c_[angle,d(("angular_projection","reset10"))],
        "joint/own_speed": np.c_[g_own,shape(unit)],
        "prefix/short_shape": np.c_[anchor[:,24:], lag(anchor[:,24:]), geom, lag(geom)],
        "prefix/three_checkpoints": np.c_[unit,lag(unit),lag(unit,2),logs,lag(logs),lag(logs,2)],
    }
    if "profile" in points[0]:
        available.update(profile_views(points,available))
    return available


def filter_scores(values, mode):
    if mode == "rise":
        out = np.full_like(values, -np.inf)
        out[...,1:] = values[...,1:]-.5*values[...,:-1]-.5*values[...,:1]
        return out
    if mode == "innovation":
        out = np.full_like(values,-np.inf)
        out[...,2:] = values[...,2:] - .5 * (values[...,1:-1] + values[...,:-2])
        return out
    return smooth(values,mode)


def score_readout(model, x):
    kind = model.get("family", "plain")
    if kind == "residual":
        gather = (x[..., -6:] - model["context_center"]) / model["context_scale"]
        expected = np.clip(gather,-3.,3.) @ np.asarray(model["context_weights"]) + model["context_intercept"]
        return score_readout(model["member"], x[..., :-6] - expected)
    if kind not in ("student", "stumps", "positive_distance"):
        return evidence(model, x)
    z = np.clip((x - model["center"]) / model["scale"], -10, 10)
    if kind == "stumps":
        score = np.zeros(z.shape[:-1])
        for k, t, low, high in model["rules"]:
            score += np.where(z[..., int(k)] <= t, low, high)
        return score
    if kind == "positive_distance":
        return -np.mean(np.square(z-model["center_positive"]),axis=-1)
    p = np.asarray(model["means"])
    s = np.asarray(model["variances"])
    a = -.5*(np.log(s)+6*np.log1p(np.square(z[...,None,:]-p)/(5*s))).mean(-1)
    return a[...,1]-a[...,0]


def transform(views, key, learner):
    x = views[key]
    if learner == "residual":
        return np.concatenate((x, views["geometry/rates"][..., :6]),axis=-1)
    return x


def score_prefix(points, bank):
    """Evaluate a single run using saved source-only members."""
    validate_steps(points)
    obs = observations(points)
    steps = np.asarray([p["step"] for p in points[1:] if p["step"] <= bank["last_step"]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if len(steps):
        x = transform(obs, bank["view"], bank["learner"])[:len(steps)]
        for item in bank["members"]:
            values = filter_scores(score_readout(item["readout"], x), bank["filter"])
            votes += values > item["cutoff"]
    alerts = steps[votes >= bank["quorum"]*len(bank["members"])]
    return {"first_alert":int(alerts[0]) if len(alerts) else None, "scored_steps":steps.tolist(), "votes":votes.tolist()}


def score_tree(points, bank):
    if "rule" not in bank:
        return score_prefix(points,bank)
    raw = [score_tree(points,member) for member in bank["members"]]
    alerts = [p["first_alert"] for p in raw if p["first_alert"] is not None]
    rule = bank["rule"]
    if rule == "either":
        first = min(alerts, default=None)
    elif rule in ("both","confirm5","confirm10"):
        duration = np.inf if rule == "both" else int(rule.removeprefix("confirm"))
        first = max(alerts) if len(alerts)==2 and abs(alerts[1]-alerts[0])<=duration else None
    else:
        raise ValueError(f"Unknown confirmation rule {rule}")
    return dict(first_alert=first,horizon=10,readouts=raw)


def score_confirmation(points, bank):
    """Apply a frozen tree to one run and optionally skip a measured no-update prefix."""
    if not bank.get("align_first_update"):
        return score_tree(points,bank)
    validate_steps(points)
    prefix = [p for p in points if p["step"]<=bank["last_step"]]
    first = next((i for i,p in enumerate(prefix[1:],1)
                  if any(p["signals"].get(f"L{layer}_magnitude",0.)>1e-8 for layer in LAYERS)), None)
    if first is None:
        return score_tree(prefix[:1],bank)
    offset = prefix[first-1]["step"]
    result = score_tree([dict(p,step=p["step"]-offset) for p in prefix[first-1:]],bank)

    def restore_clock(node):
        if node["first_alert"] is not None:
            node["first_alert"] += offset
        if "scored_steps" in node:
            node["scored_steps"] = [t+offset for t in node["scored_steps"]]
        for child in node.get("readouts",[]):
            restore_clock(child)

    restore_clock(result)
    return result
