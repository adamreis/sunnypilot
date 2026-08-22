import os
import contextlib
from openpilot.common.file_chunker import open_file_chunked, get_existing_chunks
from openpilot.common.params import Params

PARAM = "UsbGpuLoadProgress"
KERNELS_PARAM = "UsbGpuKernelTotal"
BYTE_CEIL = 30  # byte read fills 0..BYTE_CEIL, warmup fills BYTE_CEIL..100


class ProgressReader:
  def __init__(self, inner, total):
    self._inner = inner
    self._total = total
    self._params = Params()
    self._read = 0
    self._pct = -1
    self._step = max(64 * 1024, total // 100)

  def _bump(self, n):
    self._read += n
    if self._total:
      pct = min(BYTE_CEIL, self._read * BYTE_CEIL // self._total)
      if pct != self._pct:
        self._pct = pct
        self._params.put(PARAM, pct)

  def read(self, size=-1):
    data = self._inner.read(size)
    self._bump(len(data))
    return data

  def readinto(self, b):
    view = memoryview(b)
    done = 0
    while done < len(view):
      n = self._inner.readinto(view[done:done + self._step])
      if not n:
        break
      done += n
      self._bump(n)
    return done


def open_with_progress(pkl_path):
  total = sum(os.path.getsize(p) for p in get_existing_chunks(pkl_path))
  return ProgressReader(open_file_chunked(pkl_path), total)


_warmup = {"on": False, "n": 0, "total": 0, "pct": -1, "params": None}


def _install_kernel_hook():
  # runtime patch (no tinygrad source edit, so no model recompile); best-effort, never break loading
  # get_runtime runs once per kernel during graph build (each uploads a program to the eGPU = the slow warmup step)
  try:
    import tinygrad.engine.jit as jit
    if getattr(jit, "_progress_hooked", False):
      return
    orig = jit.get_runtime

    def get_runtime(*args, **kwargs):
      if _warmup["on"]:
        _warmup["n"] += 1
        if _warmup["total"]:
          pct = min(100, BYTE_CEIL + _warmup["n"] * (100 - BYTE_CEIL) // _warmup["total"])
          if pct != _warmup["pct"]:
            _warmup["pct"] = pct
            _warmup["params"].put(PARAM, pct)
      return orig(*args, **kwargs)

    jit.get_runtime = get_runtime
    jit._progress_hooked = True
  except Exception:
    pass


@contextlib.contextmanager
def warmup_progress():
  # smooth BYTE_CEIL..100 by counting eGPU kernels run during warmup; total self-calibrates across boots
  _install_kernel_hook()
  p = _warmup["params"] = Params()
  _warmup.update(on=True, n=0, pct=-1, total=p.get(KERNELS_PARAM, return_default=True))
  try:
    yield
  finally:
    _warmup["on"] = False
    if _warmup["n"]:
      p.put(KERNELS_PARAM, _warmup["n"])
    p.put(PARAM, 100)
