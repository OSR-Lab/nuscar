import importlib

from . import sm4
from . import hmac_sha1
from . import hmac_sha256
from . import hmac_sm3

__all__ = ["aes", "sm4", "hmac_sha1", "hmac_sha256", "hmac_sm3"]


def __getattr__(name):
    if name == "aes":
        module = importlib.import_module(".aes", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
