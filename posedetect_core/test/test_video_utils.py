import pytest

from posedetect_core.d2.video_utils import get_frame_count


@pytest.mark.parametrize(
    "video_path,expected_frame_count",
    [("./tests/fixtures/resources/videos/one_person_no_cuts_01/video.mp4", 73)],
)
def test_get_frame_count(video_path, expected_frame_count):
    assert expected_frame_count == get_frame_count(video_path)
