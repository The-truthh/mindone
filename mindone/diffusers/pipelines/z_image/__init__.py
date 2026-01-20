from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_transformers_available

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_output"] = ["ZImagePipelineOutput"]
_import_structure["pipeline_z_image"] = ["ZImagePipeline"]
_import_structure["pipeline_z_image_img2img"] = ["ZImageImg2ImgPipeline"]


if TYPE_CHECKING:
    from .pipeline_output import ZImagePipelineOutput
    from .pipeline_z_image import ZImagePipeline
    from .pipeline_z_image_img2img import ZImageImg2ImgPipeline

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
