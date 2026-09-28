import numpy as np

from vieneu.v3turbo import V3TurboVieNeuTTS


def _fake_tts():
    tts = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
    tts.sample_rate = 48_000
    tts.max_batch_size = 1
    tts._resolve_ref = lambda *args, **kwargs: (None, None)
    tts._infer_chunks = lambda chunks, *args: [np.ones(2, dtype=np.float32) for _ in chunks]
    tts._apply_watermark = lambda wav: wav
    tts._get_stream_scheduler = lambda: None
    tts._stream_chunks_engine = lambda chunks, *args: (
        (np.ones(2, dtype=np.float32),) for _ in chunks
    )
    return tts


def test_v3turbo_all_paths_resolve_the_same_custom_native_gaps(monkeypatch):
    captured = []

    def fake_normalize(text, max_chars):
        return ["one", "two"], ["minor", "sentence", "para"][:1]

    def fake_gaps(gaps, overrides=None):
        captured.append((list(gaps), dict(overrides or {})))
        return [0.25] * len(gaps)

    monkeypatch.setattr("vieneu.v3turbo.normalize_to_chunks_v3_with_gaps", fake_normalize)
    monkeypatch.setattr("vieneu.v3turbo.gaps_to_silence", fake_gaps)
    monkeypatch.setattr("vieneu.v3turbo.join_audio_chunks", lambda chunks, sr, silence_ps: chunks[0])
    monkeypatch.setattr("vieneu.v3turbo.pause_pad_samples", lambda *args: 0)

    tts = _fake_tts()
    tts.infer("text", minor_pause_s=0.25, sentence_pause_s=0.45, paragraph_pause_s=0.60)
    list(tts.infer_stream(
        "text", minor_pause_s=0.25, sentence_pause_s=0.45, paragraph_pause_s=0.60
    ))
    tts.infer_batch(
        ["text"], minor_pause_s=0.25, sentence_pause_s=0.45, paragraph_pause_s=0.60
    )

    expected = {"minor": 0.25, "sentence": 0.45, "para": 0.60}
    assert len(captured) == 3
    assert all(overrides == expected for _, overrides in captured)


def test_v3turbo_omitted_gap_overrides_use_native_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "vieneu.v3turbo.normalize_to_chunks_v3_with_gaps",
        lambda text, max_chars: (["one", "two"], ["sentence"]),
    )
    monkeypatch.setattr(
        "vieneu.v3turbo.gaps_to_silence",
        lambda gaps, overrides=None: captured.append(overrides) or [0.50],
    )
    monkeypatch.setattr("vieneu.v3turbo.join_audio_chunks", lambda chunks, sr, silence_ps: chunks[0])

    _fake_tts().infer("text")
    assert captured == [{}]
