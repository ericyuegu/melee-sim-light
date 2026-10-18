"""The public replay ABI runs the validator's tape without a neutral reset tick."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from peppi_py import _read_slippi

from melee_sim import _native
from melee_sim.dtypes import gamestate_dtype, terminal_dtype
from tools.validation import admission, validate_replay
from tools.validation.slpz import replay_path_for_peppi

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "replays/validation/aggregate_recent/BlondHardHippopotamus.slpz"


def test_export_tuple_and_list_metadata_are_identical() -> None:
    with replay_path_for_peppi(REPLAY) as path:
        game = _read_slippi(str(path), False)
        for bitfield in (list(game.start["bitfield"]), tuple(game.start["bitfield"])):
            game.start["bitfield"] = bitfield
            assert admission.capture_issues(game, path.read_bytes()) == []
        start_list = {**game.start, "bitfield": list(game.start["bitfield"])}
        start_tuple = {**game.start, "bitfield": tuple(game.start["bitfield"])}
        validator = validate_replay.load_native()
        first = validator.export_replay(game.frames, start_list, game.metadata)
        second = validator.export_replay(game.frames, start_tuple, game.metadata)
        assert first == second


def test_public_recorded_rng_mask_and_initial_frame() -> None:
    with replay_path_for_peppi(REPLAY) as path:
        game = _read_slippi(str(path), False)
        admission.require_admissible(game, path)
        tape = validate_replay.load_native().export_replay(game.frames, game.start, game.metadata)
    assert tape["abi"] == 1
    config_size, input_size, frame_size = (tape[key] for key in ("config_size", "input_size", "frame_size"))
    raw = np.frombuffer(tape["tape"], dtype=np.uint8)
    configs = np.tile(raw[:config_size], (2, 1))
    previous = np.tile(raw[config_size:config_size + input_size], (2, 1))
    frames = raw[config_size + input_size:].reshape(-1, frame_size)
    lib = _native.library()
    ptr = ctypes.c_void_p
    lib.msl_replay_abi_version.restype = ctypes.c_uint32
    assert lib.msl_replay_abi_version() == 1
    lib.msl_batch_reset_replay.argtypes = [ptr, ptr, ptr, ptr]
    lib.msl_batch_step_replay.argtypes = [ptr, ptr, ptr, ptr, ptr]
    observations = np.zeros(2, dtype=gamestate_dtype())
    terminals = np.zeros(2, dtype=terminal_dtype())
    batch = ptr()
    _native.check(lib.msl_batch_create(str(ROOT / "data").encode(), 2, ctypes.byref(batch)), "create")
    try:
        # Invalid replay capability is rejected before either lane resets.
        invalid = configs.copy()
        invalid[1, 25] = 2  # online_fnmsubs_zero in packed replay ABI 1.
        assert lib.msl_batch_reset_replay(batch, _native.pointer(invalid), _native.pointer(previous), None) == 1
        invalid_previous = previous.copy()
        invalid_previous[1, 12] = 4  # nml_valid in the first input player.
        assert lib.msl_batch_reset_replay(batch, _native.pointer(configs), _native.pointer(invalid_previous), None) == 1
        _native.check(lib.msl_batch_reset_replay(batch, _native.pointer(configs), _native.pointer(previous), None), "reset replay")
        for index, frame in enumerate(frames[:256]):
            rows = np.tile(frame, (2, 1))
            if index == 1:
                invalid_rows = rows.copy()
                invalid_rows[1, 4 + 12] = 4
                assert lib.msl_batch_step_replay(batch, _native.pointer(invalid_rows), None,
                                               _native.pointer(observations), _native.pointer(terminals)) == 1
                assert np.all(observations["frame_id"] == -123)
            mask = np.array([1, index % 2 == 0], dtype=np.uint8)
            held = observations[1].copy()
            _native.check(lib.msl_batch_step_replay(batch, _native.pointer(rows), _native.pointer(mask),
                                                   _native.pointer(observations), _native.pointer(terminals)), "step replay")
            assert observations[0]["frame_id"] == -123 + index
            assert observations[0]["frame_pre_random_seed"] == int.from_bytes(frame[:4], "little")
            if not mask[1]:
                assert observations[1].tobytes() == held.tobytes()
    finally:
        lib.msl_batch_destroy(batch)
