# tensor 包的公共接口
from .core import Tensor
from .utils import _unbroadcast

__all__ = ['Tensor', '_unbroadcast']