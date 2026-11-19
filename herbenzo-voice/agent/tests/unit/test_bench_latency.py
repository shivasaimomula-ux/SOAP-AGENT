"""Offline checks for the latency bench. It does not open a socket or call Google."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path


def _bench():
    path = Path(__file__).resolve().parents[2] / "scripts" / "bench_latency.py"
    spec = importlib.util.spec_from_file_location("bench_latency", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_pcm_silence_and_wav_and_raw(tmp_path):
    bench = _bench()
    silence = bench.load_pcm(None)
    assert len(silence) == int(16000 * 0.4) * 2

    raw = tmp_path / "clip.pcm"
    raw.write_bytes(b"\x01\x00" * 8)
    assert bench.load_pcm(raw) == b"\x01\x00" * 8

    wav_path = tmp_path / "clip.wav"
    with wav_path.open("wb") as raw, wave.open(raw, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x02\x00" * 4)
    assert bench.load_pcm(wav_path) == b"\x02\x00" * 4


def test_parser_defaults():
    bench = _bench()
    args = bench.build_parser().parse_args([])
    assert args.base_url == "http://127.0.0.1:8081"
    assert args.pcm is None
    assert args.timeout == 30.0
