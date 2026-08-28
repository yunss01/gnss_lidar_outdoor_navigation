import pytest

from terrain_navigation_pkg.path_validity_gate_core import (
    PathValidityGateState,
)


def make_gate():
    return PathValidityGateState(0.75, 0.75, 1.5)


def test_gate_starts_fail_safe_until_validity_arrives():
    snapshot = make_gate().evaluate(10.0)
    assert snapshot.blocked
    assert snapshot.reason == 'waiting_for_path_validity'


def test_valid_path_requires_release_hysteresis():
    gate = make_gate()
    assert gate.update(True, 10.0).blocked
    assert gate.evaluate(10.74).blocked
    snapshot = gate.evaluate(10.75)
    assert not snapshot.blocked
    assert snapshot.reason == 'path_valid'


def test_short_invalid_flicker_does_not_block_commands():
    gate = make_gate()
    gate.update(True, 10.0)
    gate.evaluate(10.75)
    assert not gate.update(False, 11.0).blocked
    assert not gate.evaluate(11.70).blocked
    snapshot = gate.update(True, 11.71)
    assert not snapshot.blocked
    assert snapshot.reason == 'path_valid'


def test_persistent_invalid_path_blocks_until_stably_valid():
    gate = make_gate()
    gate.update(True, 10.0)
    gate.evaluate(10.75)
    gate.update(False, 11.0)
    snapshot = gate.evaluate(11.75)
    assert snapshot.blocked
    assert snapshot.reason == 'path_invalid'

    assert gate.update(True, 12.0).blocked
    assert gate.evaluate(12.74).blocked
    assert not gate.evaluate(12.75).blocked


def test_stale_validity_fails_closed():
    gate = make_gate()
    gate.update(True, 10.0)
    gate.evaluate(10.75)
    assert not gate.evaluate(11.5).blocked
    snapshot = gate.evaluate(11.51)
    assert snapshot.blocked
    assert snapshot.reason == 'path_validity_stale'


@pytest.mark.parametrize(
    'values',
    [(-0.1, 0.5, 1.0), (0.5, -0.1, 1.0), (0.5, 0.5, 0.0)],
)
def test_invalid_timing_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        PathValidityGateState(*values)
