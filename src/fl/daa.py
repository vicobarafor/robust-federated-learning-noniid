from collections import OrderedDict
import torch


def state_dict_to_vector(state_dict):
    parts = []
    for _, v in state_dict.items():
        if torch.is_floating_point(v):
            parts.append(v.detach().to("cpu", dtype=torch.float32).reshape(-1))
    if not parts:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(parts, dim=0)


def subtract_state_dicts(a, b):
    out = OrderedDict()
    for k in a.keys():
        av = a[k].detach().to("cpu")
        bv = b[k].detach().to("cpu")
        if torch.is_floating_point(av):
            out[k] = av.to(torch.float32) - bv.to(torch.float32)
        else:
            out[k] = av.clone()
    return out


def add_state_dicts(a, b):
    out = OrderedDict()
    for k in a.keys():
        av = a[k].detach().to("cpu")
        bv = b[k].detach().to("cpu")
        if torch.is_floating_point(av):
            out[k] = av.to(torch.float32) + bv.to(torch.float32)
        else:
            out[k] = av.clone()
    return out


def weighted_average_states(local_states, weights, reference_state=None):
    weights = [float(w) for w in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Sum of DAA weights must be positive.")
    norm_weights = [w / weight_sum for w in weights]

    ref_state = reference_state if reference_state is not None else local_states[0]
    out = OrderedDict()

    for k, v in ref_state.items():
        v_cpu = v.detach().to("cpu")
        if torch.is_floating_point(v_cpu):
            out[k] = torch.zeros_like(v_cpu, dtype=torch.float32, device="cpu")
        else:
            out[k] = v_cpu.clone()

    for state, w in zip(local_states, norm_weights):
        for k in out.keys():
            sv = state[k].detach().to("cpu")
            if torch.is_floating_point(sv):
                out[k] += sv.to(torch.float32) * w

    return out


def drift_adaptive_average(local_states, global_state, beta=1.0, **kwargs):
    if len(local_states) == 0:
        raise ValueError("local_states must be non-empty")

    global_state_cpu = OrderedDict(
        (k, v.detach().to("cpu")) for k, v in global_state.items()
    )

    deltas = [subtract_state_dicts(local_state, global_state_cpu) for local_state in local_states]
    delta_vecs = [state_dict_to_vector(d) for d in deltas]
    mean_delta_vec = torch.stack(delta_vecs, dim=0).mean(dim=0)

    dists = torch.tensor(
        [torch.norm(dv - mean_delta_vec, p=2).item() for dv in delta_vecs],
        dtype=torch.float32
    )

    weights_t = torch.softmax(-beta * dists, dim=0)
    weights = weights_t.tolist()

    mean_delta_state = weighted_average_states(
        deltas,
        weights,
        reference_state=global_state_cpu,
    )
    new_state = add_state_dicts(global_state_cpu, mean_delta_state)

    diag = {
        "client_distances": dists.tolist(),
        "client_weights": weights,
        "weight_mean": float(weights_t.mean().item()),
        "weight_std": float(weights_t.std(unbiased=False).item()),
        "weight_min": float(weights_t.min().item()),
        "weight_max": float(weights_t.max().item()),
        "distance_mean": float(dists.mean().item()),
        "distance_std": float(dists.std(unbiased=False).item()),
        "distance_min": float(dists.min().item()),
        "distance_max": float(dists.max().item()),
    }
    return new_state, diag
