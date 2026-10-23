"""The quick 'is anyone in the video' check turns an empty video into a user-facing outcome."""

import shutil
import subprocess

import numpy as np
import pytest

from motioncoach.engine.errors import NoPersonError
from motioncoach.engine.skeleton import MEDIAPIPE33
from motioncoach.media.presence import person_presence, require_person

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


class FakeBackend:
    skeleton_name, conf_threshold = "mediapipe33", 0.5

    def __init__(self, person: bool):
        self.person = person

    def open(self):
        pass

    def close(self):
        pass

    def estimate(self, frame, ts_ms):
        assert frame.ndim == 3 and max(frame.shape[:2]) <= 640 * 2
        if not self.person:
            return None, None, None
        J = len(MEDIAPIPE33.joints)
        return np.full((J, 2), 100.0, np.float32), np.ones(J, np.float32), None


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "empty.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_empty_video_is_rejected_with_reason(clip):
    with pytest.raises(NoPersonError) as e:
        require_person(clip, FakeBackend(person=False), n=6)
    assert "未检测到人物" in str(e.value) and e.value.presence["checked"] == 6


def test_person_passes(clip):
    p = person_presence(clip, FakeBackend(person=True), n=6)
    assert p["with_person"] == 6 and p["share"] == 1.0
