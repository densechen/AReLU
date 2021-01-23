from .apl import APL
from .arelu import AReLU
from .gelu import GELU
from .maxout import Maxout
from .mixture import Mixture
from .slaf import SLAF
from .swish import Swish
from torch.nn import ReLU, ReLU6, Sigmoid, LeakyReLU, ELU, PReLU, SELU, Tanh, RReLU, CELU, Softplus


__class_dict__ = {key: var for key, var in locals().items()
                  if isinstance(var, type)}
try:
    from .pau.utils import PAU
    __class_dict__["PAU"] = PAU
except Exception:
    raise NotImplementedError("")


__version__ = "0.1.0"