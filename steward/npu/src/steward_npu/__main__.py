"""Explicit opt-in wrapper around the unchanged Steward command line."""

import os
import runpy


def main():
    if os.environ.get("STEWARD_ASSEMBLY_PROFILE") != "toy":
        raise SystemExit(
            "This renderer is validated only for STEWARD_ASSEMBLY_PROFILE=toy"
        )
    os.environ["STEWARD_RENDER_BACKEND"] = "npu_dynamic"
    os.environ.setdefault("STEWARD_NPU_MATCH_A800_DEPTH", "1")
    from .renderer.dynamic_renderer import install

    install()
    runpy.run_module("steward.cli", run_name="__main__")


if __name__ == "__main__":
    main()
