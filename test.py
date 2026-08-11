"""Test suite for the video generator.

Every test here runs offline with no model downloads, so the whole suite is a
usable pre-commit gate.  The one end-to-end render is marked ``slow``:

    pytest test.py                  # fast unit tests only
    pytest test.py -m slow          # the full render
    pytest test.py -m ''            # everything
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from video_generator import (
    CROSSFADE_SECONDS,
    MIN_SCENE_SECONDS,
    PRESETS,
    VARIANTS,
    FreeTextToVideoGenerator,
    GenerationConfig,
    Scene,
    ScriptStore,
    StoryScripter,
    VideoScript,
    _parse_scenes,
    _slugify,
    _split_frontmatter,
    _stable_seed,
    build_parser,
    capability_report,
)

STORY = (
    "The lighthouse keeper had not seen a ship in forty years. "
    "Every night he lit the lamp anyway, and every morning he wrote the date in "
    "his ledger. On the fourth of October a sail appeared on the horizon. "
    "He did not trust his eyes, so he waited until noon. "
    "The ship was still there, and it was coming closer."
)


@pytest.fixture
def config(tmp_path) -> GenerationConfig:
    """A config fully sandboxed into tmp_path, at a cheap resolution."""
    return GenerationConfig(
        preset="draft",
        resolution=(320, 180),
        fps=12,
        output_dir=tmp_path / "output",
        scripts_dir=tmp_path / "video_scripts",
        cache_dir=tmp_path / ".cache",
    )


@pytest.fixture
def generator(config):
    with FreeTextToVideoGenerator(config) as gen:
        yield gen


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_preset_defaults_are_applied():
    config = GenerationConfig(preset="standard")
    assert config.resolution == (1280, 720)
    assert config.fps == 24


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        GenerationConfig(preset="ultra")


def test_odd_resolution_rejected():
    # H.264 with yuv420p cannot encode odd dimensions; fail early, not in ffmpeg.
    with pytest.raises(ValueError, match="even"):
        GenerationConfig(resolution=(641, 480))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_turbo_presets_disable_guidance(name):
    """Distilled turbo checkpoints must run at CFG 0 with no negative prompt."""
    preset = PRESETS[name]
    if "turbo" in str(preset["model"]):
        assert preset["guidance"] == 0.0
        assert preset["negative_prompt"] == ""
        assert preset["steps"] <= 4
    else:
        assert preset["guidance"] > 1.0


def test_capability_report_keys():
    assert set(capability_report()) >= {"torch", "diffusers", "transformers", "system_tts"}


# --------------------------------------------------------------------------- #
# Script parsing
# --------------------------------------------------------------------------- #


def test_parse_well_formed_scenes():
    body = (
        "## Scene 1 — 4.0s\n\n"
        "**Visual:** a lighthouse at dusk\n\n"
        "**Narration:** He lit the lamp.\n\n"
        "## Scene 2 — 2.5s\n\n"
        "**Visual:** a sail on the horizon\n\n"
        "**Narration:** Then he waited.\n"
    )
    scenes = _parse_scenes(body)
    assert len(scenes) == 2
    assert scenes[0].visual == "a lighthouse at dusk"
    # Regression: the original regex always captured an empty narration.
    assert scenes[0].narration == "He lit the lamp."
    assert scenes[0].duration == 4.0
    assert scenes[1].narration == "Then he waited."


def test_parse_tolerates_missing_fields_and_junk():
    body = (
        "## Scene 1\n\n**Visual:** only a visual\n\n"
        "## Scene 2\n\nnothing useful here\n\n"
        "## Scene 3\n\n**Narration:** only narration\n"
    )
    scenes = _parse_scenes(body)
    # The empty block is dropped; the half-filled ones survive.
    assert [s.index for s in scenes] == [1, 3]
    assert scenes[0].narration == ""
    assert scenes[0].duration == MIN_SCENE_SECONDS


def test_parse_empty_input():
    assert _parse_scenes("") == []
    assert _parse_scenes("just some prose with no headings") == []


def test_split_frontmatter_handles_malformed_yaml():
    front, body = _split_frontmatter("---\n: : not: yaml:\n---\n\nbody text\n")
    assert front == {}
    assert "body text" in body


def test_split_frontmatter_absent():
    front, body = _split_frontmatter("# just markdown\n")
    assert front == {}
    assert body == "# just markdown\n"


# --------------------------------------------------------------------------- #
# Script store round-trip
# --------------------------------------------------------------------------- #


def test_script_markdown_round_trip(tmp_path):
    original = VideoScript(
        id="2026-08-10-test-abcd",
        title="A Story: with punctuation & symbols",
        variant="Fast-cut",
        scenes=[
            Scene(1, "a lighthouse, wide shot", "He lit the lamp.", 4.25),
            Scene(2, "a sail, close up", "Then he waited.", 2.5, overlay="LATER"),
        ],
        source_story=STORY,
        config={"preset": "draft", "seed": 7},
    )
    store = ScriptStore(tmp_path / "scripts")
    store.save(original)
    loaded = store.load(original.id)

    assert loaded.id == original.id
    assert loaded.title == original.title
    assert loaded.variant == original.variant
    assert loaded.status == original.status
    assert loaded.source_story == original.source_story
    assert loaded.config == original.config
    assert len(loaded.scenes) == 2
    assert loaded.scenes[0].narration == "He lit the lamp."
    assert loaded.scenes[0].duration == pytest.approx(4.25, abs=0.06)
    assert loaded.scenes[1].overlay == "LATER"


def test_index_is_regenerated_on_save(tmp_path):
    store = ScriptStore(tmp_path / "scripts")
    store.save(VideoScript("id-1", "First", "Cinematic", [Scene(1, "v", "n")]))
    index = (store.root / "INDEX.md").read_text()
    assert "id-1" in index and "First" in index


def test_index_is_not_loaded_as_a_script(tmp_path):
    store = ScriptStore(tmp_path / "scripts")
    store.save(VideoScript("id-1", "First", "Cinematic", [Scene(1, "v", "n")]))
    assert [s.id for s in store.list_scripts()] == ["id-1"]


def test_unreadable_script_does_not_hide_the_others(tmp_path):
    store = ScriptStore(tmp_path / "scripts")
    store.save(VideoScript("good", "Good", "Cinematic", [Scene(1, "v", "n")]))
    (store.root / "broken.md").write_text("---\n[unclosed\n")
    assert "good" in [s.id for s in store.list_scripts()]


def test_accept_demotes_sibling_candidates(tmp_path):
    store = ScriptStore(tmp_path / "scripts")
    batch = {"batch": "B1"}
    a = VideoScript("a", "A", "Cinematic", [Scene(1, "v", "n")], config=dict(batch))
    b = VideoScript("b", "B", "Fast-cut", [Scene(1, "v", "n")], config=dict(batch))
    c = VideoScript("c", "C", "Documentary", [Scene(1, "v", "n")], config={"batch": "B2"})
    for script in (a, b, c):
        store.save(script)

    store.accept(a)
    assert store.load("a").status == "accepted"
    assert store.load("b").status == "rejected"
    # A different batch is untouched.
    assert store.load("c").status == "candidate"


def test_record_render_appends_and_flips_status(tmp_path):
    store = ScriptStore(tmp_path / "scripts")
    script = VideoScript("r", "R", "Cinematic", [Scene(1, "v", "n")])
    store.save(script)
    store.record_render(script, {"output": "one.mp4", "elapsed_s": 1.0})
    store.record_render(script, {"output": "two.mp4", "elapsed_s": 2.0})

    loaded = store.load("r")
    assert loaded.status == "rendered"
    assert [r["output"] for r in loaded.renders] == ["one.mp4", "two.mp4"]


def test_load_missing_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ScriptStore(tmp_path / "scripts").load("nope")


# --------------------------------------------------------------------------- #
# Story -> scenes
# --------------------------------------------------------------------------- #


def test_sentence_split():
    assert len(StoryScripter.split_sentences(STORY)) == 5


def test_long_run_on_sentence_is_broken_up():
    run_on = " ".join(["word"] * 60) + ", and " + " ".join(["more"] * 20) + "."
    assert len(StoryScripter.split_sentences(run_on)) >= 2


def test_beat_grouping_preserves_every_sentence():
    sentences = [f"Sentence {i}." for i in range(9)]
    beats = StoryScripter.group_into_beats(sentences, 4)
    assert len(beats) == 4
    # Nothing dropped, nothing reordered.
    assert " ".join(beats) == " ".join(sentences)


def test_beat_grouping_clamps_to_sentence_count():
    assert len(StoryScripter.group_into_beats(["One.", "Two."], 8)) == 2


def test_candidates_differ_in_pacing(config):
    scripts = StoryScripter(config).generate_candidates(STORY)
    assert len(scripts) == len(VARIANTS)
    assert {s.variant for s in scripts} == set(VARIANTS)
    # The whole point of the accept step is that the options are distinguishable.
    assert len({len(s.scenes) for s in scripts}) > 1
    for script in scripts:
        assert script.scenes, "every candidate must have scenes"
        assert all(s.narration for s in script.scenes)
        assert all(s.visual for s in script.scenes)
        assert all(s.duration >= MIN_SCENE_SECONDS for s in script.scenes)


def test_candidate_ids_are_unique(config):
    scripts = StoryScripter(config).generate_candidates(STORY)
    assert len({s.id for s in scripts}) == len(scripts)


def test_candidates_share_a_batch(config):
    scripts = StoryScripter(config).generate_candidates(STORY)
    assert len({s.config["batch"] for s in scripts}) == 1


def test_empty_story_rejected(config):
    with pytest.raises(ValueError, match="empty"):
        StoryScripter(config).generate_candidates("   \n  ")


def test_keywords_drop_stopwords_and_keep_subjects():
    words = StoryScripter.keywords("The keeper had not seen a ship in forty years.")
    assert "keeper" in words and "ship" in words
    assert not {"the", "had", "not", "in"} & set(words)


def test_duration_scales_with_word_count():
    short = StoryScripter.estimate_duration("Two words.", 150)
    long = StoryScripter.estimate_duration(" ".join(["word"] * 60), 150)
    assert MIN_SCENE_SECONDS <= short < long


def test_duration_is_clamped_both_ways():
    assert StoryScripter.estimate_duration("Hi.", 150) == MIN_SCENE_SECONDS
    assert StoryScripter.estimate_duration(" ".join(["w"] * 5000), 150) <= 12.0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_slugify_truncates_on_a_word_boundary():
    slug = _slugify("A cartographer spent her life drawing coastlines")
    assert slug == "a-cartographer-spent-her-life"
    assert not slug.endswith("-")


def test_slugify_handles_degenerate_input():
    assert _slugify("") == "untitled"
    assert _slugify("!!!") == "untitled"
    # A single over-long word still has to yield something usable.
    assert len(_slugify("x" * 100)) == 32


def test_title_comes_from_the_story_not_keyword_soup():
    title = StoryScripter._derive_title(
        "A cartographer spent her life drawing coastlines she had never walked. "
        "Each map was a guess."
    )
    assert title.startswith("A cartographer spent her life")


def test_stable_seed_is_process_independent():
    # Regression: the original used hash(), which varies with PYTHONHASHSEED.
    assert _stable_seed("a lighthouse") == _stable_seed("a lighthouse")
    assert _stable_seed("a lighthouse") != _stable_seed("a sail")
    assert _stable_seed("x", salt=1) != _stable_seed("x", salt=2)


def test_scene_card_is_deterministic_for_a_seed():
    a = FreeTextToVideoGenerator.render_scene_card("a lighthouse", 42, (64, 64))
    b = FreeTextToVideoGenerator.render_scene_card("a lighthouse", 42, (64, 64))
    c = FreeTextToVideoGenerator.render_scene_card("a lighthouse", 43, (64, 64))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_image_cache_is_reused(generator):
    first = generator.generate_image("a quiet harbour", seed_salt=1)
    cached = list(generator.image_cache_dir.glob("*.png"))
    assert len(cached) == 1
    second = generator.generate_image("a quiet harbour", seed_salt=1)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert len(list(generator.image_cache_dir.glob("*.png"))) == 1


def test_image_cache_survives_cleanup(generator):
    """Regression: the cache used to live in temp_dir and be wiped every run."""
    generator.generate_image("a quiet harbour", seed_salt=1)
    generator.cleanup()
    assert list(generator.image_cache_dir.glob("*.png"))


# --------------------------------------------------------------------------- #
# Ken Burns
# --------------------------------------------------------------------------- #


def test_ken_burns_frames_are_in_bounds(generator):
    image = FreeTextToVideoGenerator.render_scene_card("test", 1, (256, 256))
    clip = generator.make_scene_clip(image, duration=4.0, seed=3, overlay_text="Caption here")
    width, height = generator.config.resolution
    for t in (0.0, 0.001, 2.0, 3.999, 4.0):
        frame = clip.frame_function(t)
        assert frame.shape == (height, width, 3)
        assert frame.dtype == np.uint8


def test_ken_burns_actually_moves(generator):
    image = FreeTextToVideoGenerator.render_scene_card("test", 5, (256, 256))
    clip = generator.make_scene_clip(image, duration=4.0, seed=5)
    assert not np.array_equal(clip.frame_function(0.0), clip.frame_function(4.0))


def test_ken_burns_handles_zero_duration(generator):
    image = FreeTextToVideoGenerator.render_scene_card("test", 1, (64, 64))
    clip = generator.make_scene_clip(image, duration=0.0, seed=1)
    assert clip.frame_function(0.0).shape[:2] == generator.config.resolution[::-1]


def test_overlay_is_confined_to_its_bounding_box(generator):
    overlay = generator.build_overlay("A caption that should sit in the lower third.")
    assert overlay is not None
    _, _, (x0, y0, box_w, box_h) = overlay
    width, height = generator.config.resolution
    assert 0 <= x0 and x0 + box_w <= width
    assert 0 <= y0 and y0 + box_h <= height
    assert y0 > height / 2, "captions belong in the lower third"


def test_overlay_of_empty_text_is_none(generator):
    assert generator.build_overlay("   ") is None


def test_overlay_clamps_to_three_lines(generator):
    overlay = generator.build_overlay(" ".join(["verylongword"] * 200))
    assert overlay is not None
    _, _, (_, _, _, box_h) = overlay
    assert box_h <= generator.config.resolution[1]


def test_cover_resize_fills_exactly():
    tall = Image.new("RGB", (100, 400))
    assert FreeTextToVideoGenerator._cover_resize(tall, (320, 180)).size == (320, 180)
    wide = Image.new("RGB", (900, 50))
    assert FreeTextToVideoGenerator._cover_resize(wide, (320, 180)).size == (320, 180)


# --------------------------------------------------------------------------- #
# Timeline maths
# --------------------------------------------------------------------------- #


def test_scene_starts_account_for_crossfade_overlap():
    durations = [5.0, 4.0, 3.0]
    fade = FreeTextToVideoGenerator._crossfade_for(durations)
    assert fade == CROSSFADE_SECONDS
    starts = FreeTextToVideoGenerator._scene_starts(durations, fade)
    # Regression: narration used to be concatenated at full scene lengths while
    # the video was shortened by one fade per cut, drifting further every scene.
    assert starts == pytest.approx([0.0, 4.6, 8.2])
    total = starts[-1] + durations[-1]
    assert total == pytest.approx(sum(durations) - 2 * fade)


def test_crossfade_shrinks_for_short_scenes():
    fade = FreeTextToVideoGenerator._crossfade_for([0.6, 0.6])
    assert 0 < fade < CROSSFADE_SECONDS


def test_crossfade_disabled_for_a_single_clip():
    assert FreeTextToVideoGenerator._crossfade_for([5.0]) == 0.0


def test_crossfade_disabled_when_it_would_be_negligible():
    assert FreeTextToVideoGenerator._crossfade_for([0.1, 0.1]) == 0.0


def test_timeline_length_and_crossfade_blending(generator):
    """The assembled clip must dissolve between scenes, not hard-cut."""
    red = Image.new("RGB", (64, 64), (255, 0, 0))
    blue = Image.new("RGB", (64, 64), (0, 0, 255))
    durations = [4.0, 4.0]
    fade = FreeTextToVideoGenerator._crossfade_for(durations)
    renderers = [
        generator.make_scene_renderer(red, 4.0, seed=1),
        generator.make_scene_renderer(blue, 4.0, seed=2),
    ]
    timeline = generator.build_timeline(renderers, durations, fade)

    assert timeline.duration == pytest.approx(sum(durations) - fade)
    starts = FreeTextToVideoGenerator._scene_starts(durations, fade)

    early = timeline.frame_function(0.5).astype(float).mean(axis=(0, 1))
    late = timeline.frame_function(timeline.duration - 0.5).astype(float).mean(axis=(0, 1))
    assert early[0] > early[2], "first scene should read red"
    assert late[2] > late[0], "last scene should read blue"

    # Mid-dissolve both scenes contribute, so neither channel dominates.
    middle = timeline.frame_function(starts[1] + fade / 2).astype(float).mean(axis=(0, 1))
    assert middle[0] > 20 and middle[2] > 20


def test_timeline_covers_its_whole_duration(generator):
    image = FreeTextToVideoGenerator.render_scene_card("t", 1, (64, 64))
    durations = [3.0, 2.5, 4.0]
    fade = FreeTextToVideoGenerator._crossfade_for(durations)
    renderers = [generator.make_scene_renderer(image, d, seed=i) for i, d in enumerate(durations)]
    timeline = generator.build_timeline(renderers, durations, fade)
    width, height = generator.config.resolution
    steps = 40
    for i in range(steps + 1):
        frame = timeline.frame_function(timeline.duration * i / steps)
        assert frame.shape == (height, width, 3)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def test_speech_returns_a_real_duration(generator, tmp_path):
    path, duration = generator.generate_speech("The ship was still there.", tmp_path / "a.wav")
    assert path.exists() and path.stat().st_size > 0
    assert duration > 0.2


def test_empty_narration_yields_silence_not_a_crash(generator, tmp_path):
    path, duration = generator.generate_speech("", tmp_path / "b.wav")
    assert path.exists()
    assert duration == MIN_SCENE_SECONDS


def test_narration_text_is_not_polluted_with_voice_tags(generator, tmp_path):
    """Regression: the original prepended '[female voice]' and the TTS read it."""
    long_path, long_seconds = generator.generate_speech(
        "The lighthouse keeper had not seen a ship in forty years.", tmp_path / "long.wav"
    )
    short_path, short_seconds = generator.generate_speech("Yes.", tmp_path / "short.wav")
    assert long_seconds > short_seconds
    assert long_path.exists() and short_path.exists()


def test_normalize_scales_to_peak():
    samples = np.array([0.1, -0.05, 0.02], dtype=np.float32)
    assert float(np.max(np.abs(FreeTextToVideoGenerator._normalize(samples)))) == pytest.approx(
        0.89, abs=1e-5
    )


def test_normalize_leaves_silence_alone():
    silence = np.zeros(16, dtype=np.float32)
    assert np.array_equal(FreeTextToVideoGenerator._normalize(silence), silence)


def test_rate_change_shortens_audio():
    samples = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)
    faster = FreeTextToVideoGenerator._apply_rate(samples, 2.0)
    assert len(faster) == pytest.approx(500, abs=2)
    assert np.array_equal(FreeTextToVideoGenerator._apply_rate(samples, 1.0), samples)


def test_music_matches_requested_duration(generator):
    clip = generator.generate_music(5.0, seed=1)
    assert clip is not None
    assert clip.duration == pytest.approx(5.0, abs=0.05)


def test_music_can_be_disabled(config):
    config.music = False
    with FreeTextToVideoGenerator(config) as gen:
        assert gen.generate_music(5.0, seed=1) is None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_cleanup_removes_temp_dir_and_is_idempotent(config):
    gen = FreeTextToVideoGenerator(config)
    (gen.temp_dir / "nested").mkdir()
    (gen.temp_dir / "nested" / "file.wav").write_bytes(b"x")
    gen.cleanup()
    assert not gen.temp_dir.exists()
    # Regression: os.rmdir() failed on subdirs, and atexit + __del__ double-fired.
    gen.cleanup()


def test_context_manager_cleans_up(config):
    with FreeTextToVideoGenerator(config) as gen:
        temp_dir = gen.temp_dir
        assert temp_dir.exists()
    assert not temp_dir.exists()


def test_missing_image_model_falls_back_rather_than_raising(generator):
    """With no diffusers installed this must still return a usable image."""
    image = generator.generate_image("anything at all")
    assert isinstance(image, Image.Image)
    assert image.size == (PRESETS["draft"]["image_size"],) * 2
    # And crucially not a black frame.
    assert np.asarray(image.convert("L")).std() > 2.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.preset == "standard"
    assert not args.self_test and not args.yes


def test_parser_rejects_unknown_preset():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--preset", "ultra"])


def test_parser_accepts_a_full_run():
    args = build_parser().parse_args(
        ["--preset", "draft", "--variant", "Fast-cut", "--seed", "9", "--yes", "--no-music"]
    )
    assert (args.preset, args.variant, args.seed, args.yes, args.no_music) == (
        "draft", "Fast-cut", 9, True, True,
    )


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_full_render_end_to_end(config, tmp_path):
    """The whole pipeline, offline, asserting the bugs that used to break it."""
    with FreeTextToVideoGenerator(config) as gen:
        script, path = gen.generate_from_story(STORY, variant="Cinematic")
        temp_dir = gen.temp_dir

    # 1. The MP4 outlives cleanup (it used to be written inside temp_dir).
    assert not temp_dir.exists()
    assert path.exists() and path.stat().st_size > 10_000
    assert path.parent == config.output_dir

    # 2. The render was recorded back into the script file, with real numbers.
    saved = ScriptStore(config.scripts_dir).load(script.id)
    assert saved.status == "rendered"
    assert len(saved.renders) == 1
    record = saved.renders[0]
    assert record["output"] == str(path)
    assert record["elapsed_s"] > 0
    # 3. Encoded length == sum(scenes) - one crossfade per cut, not the raw sum.
    expected = sum(s.duration for s in saved.scenes) - CROSSFADE_SECONDS * (len(saved.scenes) - 1)
    assert record["duration_s"] == pytest.approx(expected, abs=0.35)


@pytest.mark.slow
def test_self_test_entry_point(config):
    with FreeTextToVideoGenerator(config) as gen:
        path = gen.self_test()
    assert path.exists() and path.stat().st_size > 10_000
