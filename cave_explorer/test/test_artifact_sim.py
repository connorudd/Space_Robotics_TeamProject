"""Tests for artifact_sim. Run with:  python3 test_artifact_sim.py   (or pytest)."""
import math
import sys

try:
    from cave_explorer import artifact_sim as sim   # installed package (after sourcing)
except ImportError:
    import artifact_sim as sim                      # standalone, next to the file

ONE = [['green_crystals', 8.0, 0.0, 0.0]]


def make(**kw):
    kw.setdefault('ground_truth', ONE)
    return sim.ArtifactSimulator(**kw)


def test_visible_in_range_and_fov():
    s = make()
    ok, r = s.in_view(0, 0, 0, 8.0, 0.0)
    assert ok and abs(r - 8.0) < 1e-9


def test_too_far_too_close_behind():
    s = make()
    assert not s.in_view(-1.0, 0, 0, 8.0, 0.0)[0]   # 9 m away
    assert not s.in_view(7.0, 0, 0, 8.0, 0.0)[0]    # 1 m away
    assert not s.in_view(16.0, 0, 0, 8.0, 0.0)[0]   # behind the robot


def test_fov_edge():
    s = make()
    # 120 deg HFOV -> +-60 deg. Target at 55 deg is in, 65 deg is out.
    for deg, expect in [(55, True), (65, False), (-55, True), (-65, False)]:
        a = math.radians(deg)
        assert s.in_view(0, 0, 0, 5 * math.cos(a), 5 * math.sin(a))[0] == expect


def test_fov_wraps_with_yaw():
    s = make()
    # Robot facing -x (yaw = pi), target at -x: must be visible across the wrap.
    assert s.in_view(0, 0, math.pi, -5.0, 0.0)[0]
    assert s.in_view(0, 0, -math.pi, -5.0, 0.0)[0]


def test_ids_sequential_not_truth_order():
    gt = [['a', 5.0, 0.0, 0], ['b', 0.0, 5.0, 0]]
    s = sim.ArtifactSimulator(ground_truth=gt)
    s.observe(1.0, 0, 0, math.pi / 2, None)   # looking at b first
    s.observe(2.0, 0, 0, 0.0, None)           # then a
    rep = s.reports()
    assert [r['id'] for r in rep] == [0, 1]
    assert rep[0]['type'] == 'b' and rep[1]['type'] == 'a'


def test_nobs_and_last_seen():
    s = make()
    for t in (1.0, 2.0, 3.0):
        s.observe(t, 0, 0, 0, None)
    r = s.reports()[0]
    assert r['n_obs'] == 3 and r['last_seen'] == 3.0


def test_not_visible_not_reported():
    s = make()
    s.observe(1.0, 0, 0, math.pi, None)
    assert s.reports() == []


def test_noise_converges_and_closer_is_better():
    s = make(seed=1)
    errs = []
    for k in range(200):
        s.observe(float(k), 0, 0, 0, None)
        r = s.reports()[0]
        errs.append(math.hypot(r['x'] - 8.0, r['y'] - 0.0))
    assert errs[-1] < 0.15
    # Closer observations carry more weight than distant ones.
    assert s._sigma(2.0) < s._sigma(8.0)


def test_noise_mode_is_noisier():
    assert make(fault_mode='noise')._sigma(5.0) == 3 * make()._sigma(5.0)


def test_dropout_misses_some():
    s = make(fault_mode='dropout', dropout_prob=0.5, seed=3)
    for k in range(100):
        s.observe(float(k), 0, 0, 0, None)
    n = s.reports()[0]['n_obs']
    assert 20 < n < 80


def test_late_delays_first_report():
    s = make(fault_mode='late', late_delay=5.0)
    s.observe(10.0, 0, 0, 0, None)
    s.observe(12.0, 0, 0, 0, None)
    assert s.reports() == []
    s.observe(15.5, 0, 0, 0, None)
    assert len(s.reports()) == 1


def test_false_positive_single_ghost():
    s = make(fault_mode='false_positive', false_positive_xy=(5.0, 0.0), ground_truth=[])
    for t in range(10):
        s.observe(float(t), 0, 0, 0, None)
    rep = s.reports()
    assert len(rep) == 1 and rep[0]['type'] == 'stop_sign' and rep[0]['n_obs'] == 1


def test_unknown_mode_raises():
    try:
        sim.ArtifactSimulator(fault_mode='bogus')
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_deterministic_same_seed():
    a, b = make(seed=7), make(seed=7)
    for k in range(5):
        a.observe(float(k), 0, 0, 0, None)
        b.observe(float(k), 0, 0, 0, None)
    assert a.reports() == b.reports()


def test_blocked_by_callback():
    s = make()
    s.observe(1.0, 0, 0, 0, lambda *a: True)
    assert s.reports() == []


def grid(w, h, occ=()):
    d = [0] * (w * h)
    for (c, r) in occ:
        d[r * w + c] = 100
    return d


def test_grid_line_of_sight():
    w = h = 20
    d = grid(w, h, occ=[(10, 5)])
    # origin 0,0 res 1: wall at col 10 row 5 between (2.5,5.5) and (17.5,5.5)
    assert sim.grid_line_blocked(d, w, h, 0, 0, 1.0, 2.5, 5.5, 17.5, 5.5)
    assert not sim.grid_line_blocked(d, w, h, 0, 0, 1.0, 2.5, 7.5, 17.5, 7.5)


def test_grid_unknown_does_not_block_and_end_skipped():
    w = h = 20
    d = [-1] * (w * h)
    assert not sim.grid_line_blocked(d, w, h, 0, 0, 1.0, 2.5, 5.5, 17.5, 5.5)
    d = grid(w, h, occ=[(17, 5)])   # cell under the artefact: ignored
    assert not sim.grid_line_blocked(d, w, h, 0, 0, 1.0, 2.5, 5.5, 17.5, 5.5)


def test_grid_outside_map_ignored():
    w = h = 10
    d = grid(w, h)
    assert not sim.grid_line_blocked(d, w, h, 0, 0, 1.0, -5.0, 5.5, 20.0, 5.5)


def test_bresenham_endpoints():
    cells = sim.bresenham(0, 0, 5, 3)
    assert cells[0] == (0, 0) and cells[-1] == (5, 3)


def test_default_truth_counts():
    assert len(sim.GROUND_TRUTH) == 20
    assert sum(1 for t in sim.GROUND_TRUTH if t[0] == 'stop_sign') == 5


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('ok  ', name)
            except Exception as e:  # noqa
                fails += 1
                print('FAIL', name, repr(e))
    print('failures:', fails)
    sys.exit(1 if fails else 0)
