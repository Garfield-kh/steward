"""Bounded NPU graph cache for the service's single-image inference calls."""

import copy
import torch
import types


def cache_encoder_metadata(encoder):
    original = encoder._prepare_multilevel_features
    metadata = {}

    def prepare(instance, srcs, masks, pos_embeds):
        if instance.training or masks is None or any(m is not None for m in masks):
            return original(srcs, masks, pos_embeds)
        assert len(srcs) == instance.num_feature_levels
        shape = tuple((x.shape[-2], x.shape[-1]) for x in srcs)
        key = (shape, srcs[0].shape[0], str(srcs[0].device), torch.get_default_dtype())
        if key not in metadata:
            spatial = torch.tensor(shape, dtype=torch.long, device=srcs[0].device)
            starts = torch.cat(
                (spatial.new_zeros((1,)), spatial.prod(1).cumsum(0)[:-1])
            )
            ratios = torch.ones(
                (srcs[0].shape[0], instance.num_feature_levels, 2),
                device=srcs[0].device,
            )
            metadata[key] = (spatial, starts, ratios)
        spatial, starts, ratios = metadata[key]
        src_flat = []
        pos_flat = []
        for level, (src, pos) in enumerate(zip(srcs, pos_embeds)):
            src_flat.append(src.flatten(2).transpose(1, 2))
            pos = pos.flatten(2).transpose(1, 2)
            if instance.level_embed is not None:
                pos = pos + instance.level_embed[level].view(1, 1, -1)
            pos_flat.append(pos)
        return (
            torch.cat(src_flat, 1),
            None,
            torch.cat(pos_flat, 1),
            starts,
            ratios,
            spatial,
        )

    encoder._prepare_multilevel_features = types.MethodType(prepare, encoder)


def signature(x):
    if isinstance(x, torch.Tensor):
        return ("tensor", tuple(x.shape), str(x.dtype), str(x.device))
    if isinstance(x, dict):
        return ("dict", tuple((k, signature(v)) for k, v in sorted(x.items())))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(signature(v) for v in x))
    if hasattr(x, "__dict__"):
        return (type(x).__name__, signature(vars(x)))
    return (type(x).__name__, x)


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone(v) for v in x)
    if hasattr(x, "__dict__"):
        y = copy.copy(x)
        for k, v in vars(x).items():
            setattr(y, k, clone(v))
        return y
    return x


def update(dst, src):
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, dict):
        for k, v in src.items():
            update(dst[k], v)
    elif isinstance(src, (list, tuple)):
        for a, b in zip(dst, src):
            update(a, b)
    elif hasattr(src, "__dict__"):
        update(vars(dst), vars(src))


class GroundingGraphs:
    def __init__(self, model, max_entries=4):
        self.model = model
        self.original = model.forward_grounding
        self.max_entries = max_entries
        self.cache = {}
        self.replays = 0
        self.fallbacks = 0
        cache_encoder_metadata(model.transformer.encoder)
        from .decoder_meta import install

        install(model)

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        # Do not capture pointers to temporary autocast weight-cache entries.
        with torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False):
            return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        if (
            args
            or self.model.training
            or kwargs.get("find_target") is not None
            or kwargs["find_input"].text_ids.numel() not in (10, 11, 15, 21)
        ):
            self.fallbacks += 1
            return self.original(*args, **kwargs)
        key = signature(kwargs)
        if key not in self.cache:
            if len(self.cache) >= self.max_entries:
                self.fallbacks += 1
                return self.original(**kwargs)
            static = clone(kwargs)
            for _ in range(3):
                self.original(**static)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                out = self.original(**static)
            outputs = {
                k: out[k]
                for k in (
                    "pred_logits",
                    "presence_logit_dec",
                    "pred_boxes",
                    "pred_masks",
                )
            }
            self.cache[key] = (static, graph, outputs)
        static, graph, outputs = self.cache[key]
        update(static, kwargs)
        graph.replay()
        self.replays += 1
        return outputs
