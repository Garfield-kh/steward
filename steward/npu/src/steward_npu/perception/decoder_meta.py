"""Keep decoder dimensions as host metadata, never read NPU scalars per layer."""

import types


def install(model):
    decoder = model.transformer.decoder
    run = model._run_decoder
    rpb = decoder._get_rpb_matrix
    coord_cache = {}
    context = {"shape": None}

    def run_decoder(*args, **kwargs):
        enc = kwargs.get("encoder_out")
        old = context["shape"]
        sizes = enc.get("vis_feat_sizes") if enc is not None else None
        context["shape"] = (
            tuple(map(int, sizes[0]))
            if sizes is not None and len(sizes) == 1 and not model.training
            else None
        )
        try:
            return run(*args, **kwargs)
        finally:
            context["shape"] = old

    def rpb_host(instance, reference_boxes, feat_size):
        shape = context["shape"]
        if shape is None or instance.training:
            return rpb(reference_boxes, feat_size)
        key = (shape, str(reference_boxes.device))
        if key not in coord_cache:
            # Preserve original device-scalar division when constructing coordinate values.
            coord_cache[key] = instance._get_coords(
                feat_size[0], feat_size[1], reference_boxes.device
            )
        old_coords, old_size = (
            instance.compilable_cord_cache,
            instance.compilable_stored_size,
        )
        instance.compilable_cord_cache = coord_cache[key]
        instance.compilable_stored_size = shape
        try:
            return rpb(reference_boxes, shape)
        finally:
            instance.compilable_cord_cache = old_coords
            instance.compilable_stored_size = old_size

    model._run_decoder = run_decoder
    decoder._get_rpb_matrix = types.MethodType(rpb_host, decoder)
