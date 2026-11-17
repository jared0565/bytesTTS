# SPDX-License-Identifier: MPL-2.0
import wave
from pathlib import Path

import numpy as np
import pytest

from bytestts.audio import Audio, concat


def test_wav_roundtrip(tmp_path: Path) -> None:
    a = Audio(np.array([0.0, 0.5, -0.5, 1.5], np.float32), 22050)
    p = tmp_path / "ü dir" / "o.wav"
    p.parent.mkdir()
    a.save(p)
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (22050, 1, 2, 4)
        frames = np.frombuffer(w.readframes(4), "<i2")
    assert frames.tolist() == [0, 16384, -16384, 32767]  # 1.5 is clipped to full scale


def test_clip() -> None:
    assert np.frombuffer(Audio(np.array([2.0], np.float32), 8000).to_wav_bytes()[44:], "<i2")[0] == 32767
    assert np.frombuffer(Audio(np.array([-2.0], np.float32), 8000).to_wav_bytes()[44:], "<i2")[0] == -32768


def test_duration() -> None:
    assert Audio(np.zeros(22050, np.float32), 22050).duration == 1.0


@pytest.mark.parametrize(
    "samples",
    [np.zeros(4, np.float64), np.zeros((2, 2), np.float32), np.zeros(4, np.int16)],
)
def test_samples_must_be_1d_float32(samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Audio(samples, 22050)


@pytest.mark.parametrize("rate", [0, -1])
def test_sample_rate_must_be_positive(rate: int) -> None:
    with pytest.raises(ValueError):
        Audio(np.zeros(1, np.float32), rate)


def test_concat_pause_between_only() -> None:
    a = Audio(np.ones(10, np.float32), 100)
    out = concat([a, a], pause_s=0.1)
    assert out.samples.shape == (30,) and out.samples[10:20].max() == 0.0
    assert out.samples[:10].min() == 1.0 and out.samples[20:].min() == 1.0  # no leading or trailing pause


def test_concat_single_chunk_has_no_pause() -> None:
    a = Audio(np.ones(10, np.float32), 100)
    assert concat([a], pause_s=0.5).samples.shape == (10,)


def test_concat_rate_mismatch() -> None:
    with pytest.raises(ValueError):
        concat([Audio(np.ones(1, np.float32), 1), Audio(np.ones(1, np.float32), 2)])


def test_concat_empty_and_negative_pause() -> None:
    with pytest.raises(ValueError):
        concat([])
    with pytest.raises(ValueError):
        concat([Audio(np.ones(1, np.float32), 1)], pause_s=-0.1)
