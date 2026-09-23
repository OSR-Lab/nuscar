"""CPU feature detection for the hardware-only AES implementation."""

import ctypes
from functools import lru_cache
import platform
import subprocess


_X86_REQUIRED = ("aes", "sse2")
_AARCH64_REQUIRED = ("aes", "pmull", "neon")


def _normalize_arch(machine):
    machine = machine.lower()
    if machine in {"x86_64", "amd64", "x64", "i386", "i486", "i586", "i686", "x86"}:
        return "x86"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    return "unknown"


def _normalize_features(features):
    normalized = {feature.strip().lower() for feature in features if feature.strip()}
    if "asimd" in normalized:
        normalized.add("neon")
    return normalized


def _missing_aes_features(arch, features):
    features = _normalize_features(features)
    if arch == "x86":
        required = _X86_REQUIRED
    elif arch == "aarch64":
        required = _AARCH64_REQUIRED
    else:
        return ()
    return tuple(feature for feature in required if feature not in features)


def _x86_runtime_features(get_host_features=None):
    if get_host_features is None:
        try:
            from llvmlite.binding import get_host_cpu_features as get_host_features
        except (ImportError, OSError):
            return set()
    try:
        features = get_host_features()
        return _normalize_features(
            feature for feature, enabled in features.items() if enabled
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return set()


def _getauxval(key):
    try:
        libc = ctypes.CDLL(None)
        getauxval = libc.getauxval
        getauxval.argtypes = [ctypes.c_ulong]
        getauxval.restype = ctypes.c_ulong
        return int(getauxval(key))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def _linux_features(arch, get_x86_features=_x86_runtime_features, getauxval=_getauxval):
    if arch == "x86":
        return get_x86_features()

    if arch == "aarch64":
        hwcap = getauxval(16)
        features = set()
        if hwcap & (1 << 1):
            features.update(("asimd", "neon"))
        if hwcap & (1 << 3):
            features.add("aes")
        if hwcap & (1 << 4):
            features.add("pmull")
        return features

    return set()


def _sysctl_value(key):
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _darwin_features(arch, get_sysctl=_sysctl_value):
    if arch == "x86":
        features = set()
        for key in (
            "machdep.cpu.features",
            "machdep.cpu.leaf7_features",
            "machdep.cpu.extfeatures",
        ):
            value = get_sysctl(key)
            if value:
                features.update(value.split())
        return _normalize_features(features)

    if arch == "aarch64":
        features = set()
        if get_sysctl("hw.optional.arm.FEAT_AES") == "1":
            features.add("aes")
        if get_sysctl("hw.optional.arm.FEAT_PMULL") == "1":
            features.add("pmull")
        if get_sysctl("hw.optional.neon") == "1":
            features.add("neon")
        if get_sysctl("hw.optional.AdvSIMD") == "1":
            features.add("asimd")
        return _normalize_features(features)

    return set()


def _windows_features(arch, is_feature_present=None, get_cpu_info=None):
    if arch == "aarch64":
        if is_feature_present is None:
            try:
                is_feature_present = ctypes.windll.kernel32.IsProcessorFeaturePresent
            except (AttributeError, OSError):
                return set()

        features = set()
        try:
            if is_feature_present(19):
                features.add("neon")
            if is_feature_present(30):
                features.update(("aes", "pmull"))
        except (ctypes.ArgumentError, OSError, TypeError, ValueError):
            return set()
        return features

    if arch == "x86":
        if get_cpu_info is not None:
            try:
                info = get_cpu_info() or {}
                return _normalize_features(info.get("flags") or ())
            except (OSError, RuntimeError, TypeError, ValueError):
                return set()
        return _x86_runtime_features()

    return set()


def _detect_features(system=None, machine=None):
    system = system or platform.system()
    arch = _normalize_arch(machine or platform.machine())

    if system == "Linux":
        features = _linux_features(arch)
    elif system == "Darwin":
        features = _darwin_features(arch)
    elif system == "Windows":
        features = _windows_features(arch)
    else:
        features = set()

    return arch, features, system


def _validate_aes_features(arch, features, system):
    if arch not in {"x86", "aarch64"} or not features:
        raise RuntimeError(
            "nuscar.ciphers.aes cannot reliably detect hardware AES support "
            f"on {system}/{arch}; refusing to load hardware-only AES"
        )

    missing = _missing_aes_features(arch, features)
    if not missing:
        return

    missing_text = ", ".join(missing)
    if arch == "x86":
        raise RuntimeError(
            "nuscar.ciphers.aes requires AES-NI and SSE2; "
            f"missing: {missing_text}"
        )
    raise RuntimeError(
        "nuscar.ciphers.aes requires ARM AES, PMULL, and NEON; "
        f"missing: {missing_text}"
    )


@lru_cache(maxsize=1)
def _cached_detection():
    return _detect_features()


def require_aes_support():
    """Raise RuntimeError unless this CPU supports the native AES backend."""
    arch, features, system = _cached_detection()
    _validate_aes_features(arch, features, system)
