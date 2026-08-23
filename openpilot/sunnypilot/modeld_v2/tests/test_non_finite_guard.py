"""Regression tests for the non-finite model-output guard.

The guard exists to keep a non-finite value out of prev_feat, which feeds back
into features_buffer on the next frame. commaai/openpilot#38556 states the
intent: "we must drop non-finite model output frames while big model uses
prev_feat".

The regression these cover: the guard ran AFTER the prev_feat write and checked
only the parsed 'plan', so a NaN confined to hidden_state was never inspected --
the frame was published as good AND the recurrent state was poisoned.

Runnable without pytest (AGNOS has none):  python3 test_non_finite_guard.py
"""
import contextlib
import tempfile
import pathlib

import numpy as np

from openpilot.sunnypilot.modeld_v2.tests import helpers as H


class FakeBuf:
  def __init__(self, nbytes):
    self.data = bytearray(nbytes)


class FakeTensor:
  """Stands in for the tinygrad Tensor the JIT returns."""
  def __init__(self, arr):
    self._arr = arr

  def numpy(self):
    return self._arr


@contextlib.contextmanager
def model_state(archetype_name='supercombo_non20hz', usbgpu=True):
  """Build a real ModelState against a stubbed pkl, without pytest fixtures."""
  from openpilot.common.hardware import hw
  import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
  import openpilot.sunnypilot.models.helpers as models_helpers

  archetype = H.ARCHETYPES[archetype_name]
  with tempfile.TemporaryDirectory() as d:
    tmp = pathlib.Path(d)
    H.write_pkl(tmp, archetype)
    bundle = H.make_bundle(archetype)
    saved = (models_helpers.get_active_bundle, modeld_module.get_active_bundle, hw.Paths.model_root)
    models_helpers.get_active_bundle = lambda params=None: bundle
    modeld_module.get_active_bundle = lambda params=None: bundle
    hw.Paths.model_root = staticmethod(lambda: str(tmp))
    try:
      ms = modeld_module.ModelState(cam_w=H.CAM_W, cam_h=H.CAM_H)
      ms.usbgpu = usbgpu
      yield ms
    finally:
      models_helpers.get_active_bundle, modeld_module.get_active_bundle, hw.Paths.model_root = saved


def drive(ms, raw):
  """Push one frame through ModelState.run() with a stubbed run_policy.

  The output parser is stubbed too: this exercises the guard, not MDN parsing,
  and a synthetic raw vector is not a valid plan encoding.
  """
  ms.run_policy = lambda **kw: raw
  ms.warp = lambda **kw: None
  ms.parser.parse_outputs = lambda sliced: {
    'plan': np.zeros((1, 33, 15), dtype=np.float32),
    'desired_curvature': np.zeros((1, 1), dtype=np.float32),
  }
  bufs = {k: FakeBuf(v[3]) for k, v in ms.frame_buf_params.items()}
  transforms = {ms._road_key: np.eye(3, dtype=np.float32),
                ms._wide_key: np.eye(3, dtype=np.float32)}
  inputs = {ms.desire_key: np.zeros_like(ms.numpy_inputs[ms.desire_key]),
            'traffic_convention': np.zeros((1, 2), dtype=np.float32),
            'lateral_control_params': np.zeros((1, 2), dtype=np.float32)}
  return ms.run(bufs, transforms, inputs, prepare_only=False)


def supercombo_output(hidden_state_nan=False, plan_nan=False):
  out = np.full(1062, 0.01, dtype=np.float32)
  if plan_nan:
    out[H.SUPERCOMBO_SLICES['plan'].start] = np.nan
  if hidden_state_nan:
    out[H.SUPERCOMBO_SLICES['hidden_state'].start] = np.nan
  return FakeTensor(out.reshape(1, -1))


# --- scenarios (plain functions so they run with or without pytest) ----------

def scenario_non_finite_never_reaches_prev_feat(nan_in):
  """A non-finite value ANYWHERE in the raw output must not be latched.

  'hidden_state' is the regression: the old guard inspected only the parsed
  'plan', so this frame passed and poisoned the recurrent state.
  """
  with model_state() as ms:
    assert 'prev_feat' in ms.numpy_inputs, 'archetype has no host-side prev_feat'
    ms.numpy_inputs['prev_feat'][:] = 0.0

    out = drive(ms, supercombo_output(hidden_state_nan=(nan_in == 'hidden_state'),
                                      plan_nan=(nan_in == 'plan')))

    assert out is None, f'frame with NaN in {nan_in} must be dropped'
    assert np.all(np.isfinite(ms.numpy_inputs['prev_feat'])), \
      f'NaN in {nan_in} leaked into prev_feat -- recurrent state poisoned'
    assert np.all(ms.numpy_inputs['prev_feat'] == 0.0), \
      'prev_feat must retain its last-good value'


def scenario_recovers_on_next_clean_frame():
  """Dropping a frame must not wedge the model."""
  with model_state() as ms:
    assert drive(ms, supercombo_output(hidden_state_nan=True)) is None
    assert drive(ms, supercombo_output()) is not None, \
      'model did not recover on the next clean frame'


def scenario_guard_inactive_without_usbgpu():
  """The guard is usbgpu-only; the on-device path must be unaffected."""
  with model_state(usbgpu=False) as ms:
    assert drive(ms, supercombo_output(hidden_state_nan=True)) is not None


# --- pytest entry points -----------------------------------------------------

def test_non_finite_in_hidden_state_never_reaches_prev_feat():
  scenario_non_finite_never_reaches_prev_feat('hidden_state')


def test_non_finite_in_plan_never_reaches_prev_feat():
  scenario_non_finite_never_reaches_prev_feat('plan')


def test_recovers_on_next_clean_frame():
  scenario_recovers_on_next_clean_frame()


def test_guard_inactive_without_usbgpu():
  scenario_guard_inactive_without_usbgpu()


if __name__ == '__main__':
  import sys
  cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
  failed = 0
  for name, fn in cases:
    try:
      fn()
      print(f'  PASS  {name}')
    except AssertionError as e:
      failed += 1
      print(f'  FAIL  {name}: {e}')
    except Exception as e:
      failed += 1
      print(f'  ERROR {name}: {type(e).__name__}: {e}')
  print(f'{len(cases) - failed}/{len(cases)} passed')
  sys.exit(1 if failed else 0)
