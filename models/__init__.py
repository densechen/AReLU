from .models import BaseModel
from .conv import ConvMNIST
from .linear import LinearMNIST


__class_dict__ = {key: var for key, var in locals().items()
                  if isinstance(var, type)}
