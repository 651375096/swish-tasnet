import pathlib

# from .models import ConvTasNet, DCCRNet, DCUNet, DPRNNTasNet, DPTNet, LSTMTasNet, DeMask
from .models import ConvTasNet
from .utils import deprecation_utils, torch_utils  # noqa

project_root = str(pathlib.Path(__file__).expanduser().absolute().parent.parent)
__version__ = "0.5.2"


def show_available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    print(" \n".join(list(MODELS_URLS_HASHTABLE.keys())))


def available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    return MODELS_URLS_HASHTABLE

#
# __all__ = [
#     "ConvTasNet",
#     "DPRNNTasNet",
#     "DPTNet",
#     "LSTMTasNet",
#     "DeMask",
#     "DCUNet",
#     "DCCRNet",
#     "show_available_models",
# ]

__all__ = [
    "ConvTasNet",
    # "DPRNNTasNet",
    # "DPTNet",
    # "LSTMTasNet",
    # "DeMask",
    # "DCUNet",
    # "DCCRNet",
    # "show_available_models",
]
