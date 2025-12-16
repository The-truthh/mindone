from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["decoders"] = ["WanImageVaeDecoderStep"]
_import_structure["encoders"] = ["WanTextEncoderStep"]
_import_structure["modular_blocks"] = [
    "ALL_BLOCKS",
    "Wan22AutoBlocks",
    "WanAutoBlocks",
    "WanAutoImageEncoderStep",
    "WanAutoVaeImageEncoderStep",
]
_import_structure["modular_pipeline"] = ["WanModularPipeline"]

if TYPE_CHECKING:
    from .decoders import WanImageVaeDecoderStep
    from .encoders import WanTextEncoderStep
    from .modular_blocks import (
        ALL_BLOCKS,
        Wan22AutoBlocks,
        WanAutoBlocks,
        WanAutoImageEncoderStep,
        WanAutoVaeImageEncoderStep,
    )
    from .modular_pipeline import WanModularPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
