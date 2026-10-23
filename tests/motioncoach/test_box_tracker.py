"""Single-target person box tracking for top-down pose models (perception.base.BoxTracker)."""

import numpy as np

from motioncoach.perception.base import BoxTracker

SIZE = (1080, 1920)
PERSON = np.array([400.0, 300.0, 700.0, 1500.0])
BYSTANDER = np.array([850.0, 400.0, 1050.0, 1400.0])


def test_follows_the_detection_and_bridges_a_missed_frame_from_the_skeleton():
    tr = BoxTracker(max_bridge_s=1.0)
    box, src = tr.choose(np.array([PERSON, BYSTANDER]), 0)
    assert src == "detected" and np.allclose(box, PERSON)            # largest detection starts the target
    tr.update(box, PERSON + [20, 40, -20, -10], src, 0, SIZE)
    # deep bend: the detector sees only the bystander -> keep the target's box, do not jump
    box, src = tr.choose(np.array([BYSTANDER]), 100)
    assert src == "bridged" and box[0] < 600 and box[2] < 850
    bent = np.array([380.0, 700.0, 720.0, 1500.0])                   # the skeleton found in the bridged box
    tr.update(box, bent, src, 100, SIZE)
    box, src = tr.choose(np.zeros((0, 4)), 200)
    assert src == "bridged" and box[1] > 600                          # the prediction follows the body down
    assert tr.stats["detected"] == 1 and tr.stats["bridged"] == 1


def test_gives_up_bridging_after_the_limit_and_reacquires():
    tr = BoxTracker(max_bridge_s=1.0)
    box, src = tr.choose(np.array([PERSON]), 0)
    tr.update(box, PERSON, src, 0, SIZE)
    for ms in (100, 500, 900):                                        # nothing confident while bridging
        box, src = tr.choose(np.zeros((0, 4)), ms)
        assert src == "bridged"
        tr.update(box, None, src, ms, SIZE)
    box, src = tr.choose(np.array([BYSTANDER]), 1200)                 # > 1 s since the last good estimate
    assert src == "detected" and np.allclose(box, BYSTANDER)
    assert tr.stats["lost"] == 3
