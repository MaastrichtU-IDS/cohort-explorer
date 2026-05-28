import ctypes
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_OVERWRITE_PASSES = 3

_OVERWRITE_PATTERNS = [0x55, 0xAA, 0x00]

def _try_mlock(buf: bytearray) -> bool:
    \
\
\
\
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            addr = (ctypes.c_char * len(buf)).from_buffer(buf)
            result = libc.mlock(ctypes.addressof(addr), len(buf))
            return result == 0
        except (OSError, AttributeError):
            return False
    return False

def _try_munlock(buf: bytearray) -> None:
    \
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            addr = (ctypes.c_char * len(buf)).from_buffer(buf)
            libc.munlock(ctypes.addressof(addr), len(buf))
        except (OSError, AttributeError):
            pass

def secure_zero(buf: bytearray) -> None:
    \
\
\
\
\
\
\
    if not isinstance(buf, bytearray):
        raise TypeError("secure_zero requires a mutable bytearray, not bytes")

    if len(buf) == 0:
        return

    for pattern in _OVERWRITE_PATTERNS:
        ctypes.memset(
            (ctypes.c_char * len(buf)).from_buffer(buf),
            pattern,
            len(buf),
        )

    for i in range(len(buf)):
        if buf[i] != 0:

            raise RuntimeError("secure_zero: verification failed")

class SecureKeyBuffer:
    \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
    __slots__ = ('_buf', '_locked', '_erased')

    def __init__(self, key_material: bytes | bytearray):
        self._buf = bytearray(key_material)
        self._erased = False
        self._locked = _try_mlock(self._buf)
        if not self._locked:
            logger.debug(
                "Could not mlock key buffer. Key material may be swapped "
                "to disk. Use an HSM for production deployments."
            )

        if isinstance(key_material, bytearray):
            secure_zero(key_material)

    def read(self) -> bytes:
        \
        if self._erased:
            raise RuntimeError("Key buffer has been securely erased")
        return bytes(self._buf)

    def evolve(self, new_key: bytes | bytearray) -> None:
        \
\
\
\
\
\
\
\
        if self._erased:
            raise RuntimeError("Key buffer has been securely erased")

        secure_zero(self._buf)
        if self._locked:
            _try_munlock(self._buf)

        self._buf = bytearray(new_key)
        self._locked = _try_mlock(self._buf)

        if isinstance(new_key, bytearray):
            secure_zero(new_key)

    def erase(self) -> None:
        \
        if not self._erased:
            secure_zero(self._buf)
            if self._locked:
                _try_munlock(self._buf)
            self._erased = True

    @property
    def is_erased(self) -> bool:
        return self._erased

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.erase()

    def __del__(self):
        \
        if not self._erased:
            try:
                self.erase()
            except Exception:
                pass

    def __repr__(self):
        if self._erased:
            return "SecureKeyBuffer(ERASED)"
        return f"SecureKeyBuffer({len(self._buf)} bytes, locked={self._locked})"
