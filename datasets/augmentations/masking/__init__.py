from datasets.augmentations.masking.deep_fake_mask import ComponentsMasking, DFLMasking, ExtendedMasking
from datasets.augmentations.masking.grid import GridMasking
from datasets.augmentations.masking.meshgrid import MeshgridMasking
from datasets.augmentations.masking.sladd import SladdMasking
from datasets.augmentations.masking.whole import WholeMasking


def from_name(name: str, **kwargs):
    if name == "grid":
        return GridMasking(**kwargs)
    elif name == "sladd":
        return SladdMasking(**kwargs)
    elif name == "meshgrid":
        return MeshgridMasking(**kwargs)
    elif name == "whole":
        return WholeMasking(**kwargs)
    elif name == "dfl":
        return DFLMasking(**kwargs)
    elif name == "components":
        return ComponentsMasking(**kwargs)
    elif name == "extended":
        return ExtendedMasking(**kwargs)
    else:
        raise NotImplementedError
