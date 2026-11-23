"""Test suite for the film generator.

Everything here runs offline with no model downloads, so the whole suite works
as a pre-commit gate. The single end-to-end render is marked ``slow``:

    pytest test.py              # fast unit tests only
    pytest test.py -m slow      # the full render
    pytest test.py -m ''        # everything

`MockBackend` stands in for the diffusion stack, which cannot run on this
machine at all. That is what lets the identity, motion and capability-degrade
paths be tested here despite no model ever being loaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_gen import (
    ASPECTS,
    FORMATS,
    PRESETS,
    STYLE_TIERS,
    Character,
    CharacterStore,
    FilmRenderer,
    GenerationConfig,
    Interview,
    Line,
    Production,
    ProductionStore,
    Scene,
    Screenwriter,
    Shot,
    SpeechSynthesizer,
    StoryMeta,
    VoiceProfile,
    cast_production,
    infer_meta,
)
from video_gen.casting import SystemVoice, cast_voice, infer_traits
from video_gen.config import CROSSFADE_SECONDS, MIN_SHOT_SECONDS
from video_gen.llm import (
    LLMError,
    LLMReply,
    MockLLM,
    OllamaClient,
    OllamaModel,
    discover_llm,
    extract_json,
)
from video_gen.llm import probe as llm_probe
from video_gen.writersroom import (
    BRIEF_MAX_WORDS,
    MAX_LINE_WORDS,
    MIN_DRAFT_LINES,
    WritersRoom,
    clean_line,
    echoes,
    resolve_speaker,
    validate_cast,
    validate_lines,
)
from video_gen.model import NARRATOR, character_key, parse_scenes, slugify, stable_seed
from video_gen.music import MOODS, duck, mood_for, render_bed, score_timeline
from video_gen import agent, providers
from video_gen.agent import (
    EXIT_CANCELLED,
    EXIT_FAILED,
    EXIT_NO_AUDIO,
    EXIT_OK,
    EXIT_USAGE,
    CommandResult,
)
from video_gen.casting import discover_espeak_voices, _parse_espeak_rows
from video_gen.score import (
    BED,
    CUE,
    PROMPT,
    HostedMusic,
    MockMusic,
    MusicRegistry,
    ProceduralMusic,
    ScoreRequest,
    ScoreResult,
    decode_audio,
)
from video_gen.speech import apply_pitch, apply_rate, normalize
from video_gen.visuals import (
    IDENTITY,
    MOTION,
    STILL,
    TRAIN,
    BackendRegistry,
    CharacterRef,
    MockBackend,
    ProceduralBackend,
    ShotRequest,
    character_palette,
)
from video_gen.visuals.base import ShotVisual

SCRIPT = """INT. CONTROL ROOM - NIGHT

Banks of monitors glow in the dark.

MAYA: The signal is back. Same rhythm as last night.
JONAS (wary): Then it was never noise.
MAYA: We are out of time.

EXT. CAR PARK - DAWN

Rain falls on empty tarmac.

JONAS: Drive carefully.
"""

PROSE = (
    "Maya had run the night shift for three years. The signal came at 3am, a rhythm "
    'buried in the static. "We are out of time," she said, watching the trace climb. '
    'Jonas leaned over the console and frowned. "Then we go tonight," Jonas replied. '
    "Outside, the rain kept falling on the empty car park."
)


@pytest.fixture
def config(tmp_path) -> GenerationConfig:
    """Config fully sandboxed into tmp_path, at a cheap resolution."""
    return GenerationConfig(
        preset="draft",
        resolution=(320, 180),
        fps=12,
        output_dir=tmp_path / "output",
        scripts_dir=tmp_path / "video_scripts",
        characters_dir=tmp_path / "characters",
        cache_dir=tmp_path / ".cache",
    )


@pytest.fixture
def renderer(config):
    registry = BackendRegistry([ProceduralBackend(config)])
    with FilmRenderer(config, registry=registry) as film:
        yield film


def sample_production() -> Production:
    production = Production(id="t1", title="The Signal", logline="Two engineers hear something.")
    production.add_character(
        Character("Maya", role="engineer", description="A sharp woman. She runs the night shift.")
    )
    production.add_character(Character("Jonas", role="technician", description="An older man."))
    scene = Scene(1, setting="Control room", time_of_day="night", mood="tense")
    scene.shots = [
        Shot(1, "establishing", ["Maya"], action="monitors glow", camera="push in",
             visual="a dim control room", lines=[Line(NARRATOR, "It began at 3am.")], duration=4.0),
        Shot(2, "close", ["Maya"], lines=[Line("Maya", "We're out of time.", "whispering")],
             duration=3.2),
        Shot(3, "medium", ["Jonas"], lines=[Line("Jonas", "Then we go tonight.")], duration=2.8),
    ]
    production.scenes = [scene]
    return production


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_resolution_derives_from_preset_and_aspect():
    assert GenerationConfig(preset="standard", aspect="16:9").resolution == (1280, 720)
    assert GenerationConfig(preset="standard", aspect="9:16").resolution == (720, 1280)
    assert GenerationConfig(preset="standard", aspect="1:1").resolution == (720, 720)


def test_portrait_is_detected():
    assert GenerationConfig(aspect="9:16").is_portrait
    assert not GenerationConfig(aspect="16:9").is_portrait


@pytest.mark.parametrize("aspect", sorted(ASPECTS))
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_resolution_is_h264_encodable(preset, aspect):
    # H.264 with yuv420p cannot encode odd dimensions.
    width, height = GenerationConfig.resolution_for(preset, aspect)
    assert width % 2 == 0 and height % 2 == 0


def test_bad_config_values_are_rejected():
    for kwargs in ({"preset": "ultra"}, {"aspect": "3:7"}, {"style": "claymation"}):
        with pytest.raises(ValueError):
            GenerationConfig(**kwargs)
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


def test_every_style_and_format_is_complete():
    for style in STYLE_TIERS.values():
        assert style["prompt_suffix"] and style["label"]
    for spec in FORMATS.values():
        assert spec["target_seconds"] > 0 and spec["wpm"] > 0


# --------------------------------------------------------------------------- #
# Screenplay model and markdown
# --------------------------------------------------------------------------- #


def test_production_markdown_round_trip():
    original = sample_production()
    original.cast[0].voice = VoiceProfile("system", "Samantha", 0.97, -1.2, "female")
    restored = Production.from_markdown(original.to_markdown())

    assert restored.title == original.title
    assert [c.name for c in restored.cast] == ["Maya", "Jonas"]
    assert restored.character("maya").voice.voice == "Samantha"
    assert len(restored.shots) == 3
    assert restored.shots[0].camera == "push in"
    assert restored.shots[0].lines[0].is_narration
    assert restored.shots[1].lines[0].delivery == "whispering"
    assert restored.shots[1].lines[0].text == "We're out of time."
    assert restored.shots[1].duration == pytest.approx(3.2, abs=0.06)
    assert restored.scenes[0].time_of_day == "night"
    assert restored.scenes[0].mood == "tense"
    # Stable enough to re-save without churning the file.
    assert restored.to_markdown() == original.to_markdown()


def test_dialogue_and_named_fields_are_not_confused():
    """`**With:**` is metadata; `**MAYA:**` is speech. Both are bold-colon."""
    shots = parse_scenes(
        "## Scene 1 — Lab\n\n### Shot 1 — close — 3.0s\n\n"
        "**With:** Maya, Jonas\n**Camera:** push in\n**Action:** she turns\n"
        "**Visual:** a lab\n**MAYA (quiet):** Look at this.\n"
    )[0].shots
    shot = shots[0]
    assert shot.subjects == ["Maya", "Jonas"]
    assert shot.camera == "push in"
    assert shot.action == "she turns"
    assert [(l.character, l.delivery, l.text) for l in shot.lines] == [
        ("MAYA", "quiet", "Look at this.")
    ]


def test_parsing_tolerates_junk():
    assert parse_scenes("") == []
    assert parse_scenes("just prose, no headings") == []
    # A malformed shot is skipped without taking its scene down.
    scenes = parse_scenes(
        "## Scene 1 — Lab\n\n### Shot 1\n\nnothing useful\n\n### Shot 2\n\n**Action:** fine\n"
    )
    assert len(scenes) == 1 and [s.index for s in scenes[0].shots] == [2]


def test_character_key_is_case_and_space_insensitive():
    assert character_key(" maya  smith ") == character_key("MAYA SMITH")
    production = sample_production()
    assert production.character("MAYA") is production.character("maya")


def test_add_character_is_idempotent():
    production = sample_production()
    before = len(production.cast)
    production.add_character(Character("MAYA"))
    assert len(production.cast) == before


def test_speaking_cast_excludes_silent_members():
    production = sample_production()
    production.add_character(Character("Extra"))
    assert [c.name for c in production.speaking_cast] == ["Maya", "Jonas"]


def test_slugify_truncates_on_a_word_boundary():
    assert slugify("A cartographer spent her life drawing coastlines") == "a-cartographer-spent-her-life"
    assert slugify("") == "untitled" and slugify("!!!") == "untitled"
    assert len(slugify("x" * 100)) == 32


def test_stable_seed_is_process_independent():
    # Regression: hash() varies with PYTHONHASHSEED between runs.
    assert stable_seed("a lighthouse") == stable_seed("a lighthouse")
    assert stable_seed("a lighthouse") != stable_seed("a sail")
    assert stable_seed("x", 1) != stable_seed("x", 2)


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #


def test_production_store_round_trip_and_index(config):
    store = ProductionStore(config.scripts_dir)
    store.save(sample_production())
    assert store.load("t1").title == "The Signal"
    assert "t1" in (store.root / "INDEX.md").read_text()
    assert [p.id for p in store.list_productions()] == ["t1"]  # INDEX.md is not a production


def test_unreadable_file_does_not_hide_the_others(config):
    store = ProductionStore(config.scripts_dir)
    store.save(sample_production())
    (store.root / "broken.md").write_text("---\n[unclosed\n")
    assert "t1" in [p.id for p in store.list_productions()]


def test_accept_demotes_batch_siblings(config):
    store = ProductionStore(config.scripts_dir)
    keep, drop, other = sample_production(), sample_production(), sample_production()
    keep.id, drop.id, other.id = "a", "b", "c"
    keep.config = drop.config = {"batch": "B1"}
    other.config = {"batch": "B2"}
    for production in (keep, drop, other):
        store.save(production)

    store.accept(keep)
    assert store.load("a").status == "accepted"
    assert store.load("b").status == "rejected"
    assert store.load("c").status == "draft"  # a different batch is untouched


def test_record_render_appends(config):
    store = ProductionStore(config.scripts_dir)
    production = sample_production()
    store.save(production)
    store.record_render(production, {"output": "one.mp4"})
    store.record_render(production, {"output": "two.mp4"})
    restored = store.load("t1")
    assert restored.status == "rendered"
    assert [r["output"] for r in restored.renders] == ["one.mp4", "two.mp4"]


def test_missing_records_raise(config):
    with pytest.raises(FileNotFoundError):
        ProductionStore(config.scripts_dir).load("nope")
    with pytest.raises(FileNotFoundError):
        CharacterStore(config.characters_dir).load("nobody")


def test_character_store_round_trip(config):
    store = CharacterStore(config.characters_dir)
    maya = Character("Maya", "engineer", "Sharp.", "red hair", VoiceProfile("system", "Kathy", 1.02, -2.0))
    store.save(maya)
    restored = store.load(maya.id)
    assert restored.name == "Maya" and restored.appearance == "red hair"
    assert restored.voice.voice == "Kathy" and restored.voice.pitch == pytest.approx(-2.0)
    assert store.find("MAYA") is not None


def test_stored_character_fills_gaps_but_never_overrides(config):
    """A production that states its own details keeps them."""
    store = CharacterStore(config.characters_dir)
    store.save(Character("Maya", "engineer", "Stored description.", "stored hair",
                         VoiceProfile("system", "Kathy")))
    production = Production(id="p", title="T")
    production.add_character(Character("Maya", appearance="scripted hair"))
    store.merge_into(production)

    member = production.character("Maya")
    assert member.appearance == "scripted hair"  # the script wins
    assert member.description == "Stored description."  # the gap is filled
    assert member.voice.voice == "Kathy"


def test_import_reference_copies_the_image(config, tmp_path):
    source = tmp_path / "face.png"
    Image.new("RGB", (16, 16), (200, 40, 40)).save(source)
    store = CharacterStore(config.characters_dir)
    member = store.import_reference(Character("Maya"), source)
    assert member.reference_image and Path(member.reference_image).exists()
    assert Path(member.reference_image).parent == config.characters_dir
    assert store.load("maya").reference_image == member.reference_image


# --------------------------------------------------------------------------- #
# Screenwriting
# --------------------------------------------------------------------------- #


def test_script_input_is_detected_and_prose_is_not():
    assert Screenwriter.looks_like_script(SCRIPT)
    assert not Screenwriter.looks_like_script(PROSE)


def test_script_parses_into_scenes_shots_and_cast():
    production = Screenwriter().write(SCRIPT, StoryMeta(format="short"))
    assert [c.name for c in production.cast] == ["Maya", "Jonas"]
    assert len(production.scenes) == 2
    assert production.scenes[0].setting == "Control Room"
    # The slugline is the authority on time of day, not the prose below it.
    assert production.scenes[0].time_of_day == "night"
    assert production.scenes[1].time_of_day == "dawn"
    assert any(l.delivery == "wary" for s in production.shots for l in s.lines)


def test_logline_skips_screenplay_furniture():
    # Regression: naive splitting made the logline "INT.".
    assert Screenwriter().write(SCRIPT, StoryMeta()).logline.startswith("Banks of monitors")


def test_each_shot_has_at_most_one_speaker():
    """Otherwise the camera can never cut between two people talking."""
    for source in (SCRIPT, PROSE):
        for shot in Screenwriter().write(source, StoryMeta()).shots:
            speakers = {l.key for l in shot.lines if not l.is_narration}
            assert len(speakers) <= 1


def test_prose_dialogue_is_attributed_through_pronouns():
    """Regression: `"...," she said` used to be credited to the wrong character."""
    production = Screenwriter().write(PROSE, StoryMeta(format="short"))
    spoken = {l.character.upper(): l.text for s in production.shots for l in s.lines}
    assert "We are out of time." in spoken.get("MAYA", "")
    assert "Then we go tonight." in spoken.get("JONAS", "")


def test_narration_does_not_repeat_the_dialogue_or_its_attribution():
    production = Screenwriter().write(PROSE, StoryMeta(format="short"))
    narration = " ".join(l.text for s in production.shots for l in s.lines if l.is_narration)
    assert "out of time" not in narration
    assert "she said" not in narration and "Jonas replied" not in narration


def test_protagonist_named_once_is_still_cast():
    """A short story often names its lead exactly once."""
    names = {c.name for c in Screenwriter().extract_characters(PROSE)}
    assert {"Maya", "Jonas"} <= names


def test_common_nouns_are_not_mistaken_for_characters():
    text = "Banks of monitors glow. The rain fell. Outside, the rain kept falling."
    names = {c.name for c in Screenwriter().extract_characters(text)}
    assert not ({"Banks", "Rain", "Outside", "The"} & names)


BRIEF = (
    "A 20 second vertical advert for Kestrel Coffee. Ama, a barista, tells a tired "
    "commuter named Dev that the first cup is on the house."
)


def test_brand_names_are_not_cast_as_characters():
    """Regression: "Kestrel Coffee" used to become two speaking characters."""
    names = {c.name for c in Screenwriter().extract_characters(BRIEF)}
    assert names == {"Ama", "Dev"}


def test_introduced_characters_outrank_bare_capitals():
    names = {c.name for c in Screenwriter().extract_characters("A film about a woman named Ada.")}
    assert "Ada" in names


def test_reported_speech_becomes_real_dialogue():
    """A brief describes speech; the narrator must not read the brief aloud."""
    production = Screenwriter().write(BRIEF, StoryMeta(format="ad"))
    spoken = [(l.character.upper(), l.text) for s in production.shots for l in s.lines]
    assert ("AMA", "The first cup is on the house.") in spoken
    narration = " ".join(l.text for s in production.shots for l in s.lines if l.is_narration)
    assert "advert" not in narration and "tells a tired commuter" not in narration


def test_reported_speech_handles_the_plain_form():
    lines = Screenwriter().reported_speech("Ama says the coffee is free today.", ["Ama"])
    assert lines and lines[0][0] == "Ama"
    assert lines[0][1] == "The coffee is free today."


def test_reported_speech_ignores_unknown_speakers():
    assert Screenwriter().reported_speech("Bob tells Ann that it is fine.", ["Ama"]) == []


def test_style_directives_are_not_narrated():
    from video_gen.screenwriter import _is_directive

    assert _is_directive("Warm and friendly, anime style.")
    assert _is_directive("Cinematic realistic tone.")
    assert not _is_directive("Maya had run the night shift for three years.")


def test_title_strips_a_brief_preamble():
    # Regression: the title used to be the whole instruction sentence.
    assert Screenwriter.derive_title(BRIEF) == "Kestrel Coffee"
    assert Screenwriter.derive_title("Maya had run the night shift.") == "Maya had run the night shift"


def test_shot_framing_varies():
    production = Screenwriter().write(SCRIPT, StoryMeta(format="short"))
    assert production.shots[0].shot_type == "establishing"
    assert len({s.shot_type for s in production.shots}) > 1


def test_every_shot_gets_a_visual_prompt_carrying_the_style():
    production = Screenwriter().write(SCRIPT, StoryMeta(style="3d"))
    suffix_word = str(STYLE_TIERS["3d"]["prompt_suffix"]).split(",")[0]
    for shot in production.shots:
        assert shot.visual and suffix_word in shot.visual


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        Screenwriter().write("   \n  ")


def test_format_changes_pacing():
    ad = Screenwriter().write(PROSE, StoryMeta(format="ad", target_seconds=30))
    doc = Screenwriter().write(PROSE, StoryMeta(format="educational", target_seconds=120))
    assert ad.estimated_duration != doc.estimated_duration


def test_beats_never_drop_or_reorder_sentences():
    sentences = [f"Sentence {i}." for i in range(9)]
    groups = Screenwriter._group(sentences, 4)
    assert len(groups) == 4 and " ".join(groups) == " ".join(sentences)
    assert len(Screenwriter._group(["One.", "Two."], 8)) == 2


# --------------------------------------------------------------------------- #
# Interview
# --------------------------------------------------------------------------- #


def test_inference_reads_the_prompt():
    meta = infer_meta("A funny 45 second vertical advert for a bike shop, anime style, for teens")
    assert meta.format == "ad"
    assert meta.aspect == "9:16"
    assert meta.style == "2d"
    assert meta.target_seconds == 45
    assert meta.audience == "teens"
    assert meta.tone == "playful"


def test_inference_understands_minutes():
    assert infer_meta("a 2 minute explainer").target_seconds == 120


def test_inference_falls_back_to_format_defaults():
    meta = infer_meta("a short film about a lighthouse")
    assert meta.target_seconds == 0
    assert meta.resolved_target() == FORMATS["short"]["target_seconds"]


def test_interview_is_a_no_op_when_non_interactive():
    meta = Interview(ask=None).run("a short film about a lighthouse")
    assert meta.format == "short"


def test_interview_answers_are_applied_and_coerced():
    meta = infer_meta("x")
    Interview._apply(meta, {"format": "ad", "aspect": "9:16", "target_seconds": "20 seconds",
                            "cast": "Ana, Bo", "style": "3d"})
    assert (meta.format, meta.aspect, meta.style) == ("ad", "9:16", "3d")
    assert meta.target_seconds == 20
    assert meta.character_names == ["Ana", "Bo"]


def test_interview_ignores_nonsense_answers():
    meta = infer_meta("a short film")
    Interview._apply(meta, {"format": "not-a-format", "aspect": "banana"})
    assert meta.format == "short" and meta.aspect == "16:9"


def test_explicit_overrides_beat_inference_and_are_never_asked():
    asked = []

    def ask(prompt, default):
        asked.append(prompt)
        return default

    meta = Interview(ask=ask).run("a vertical tiktok advert", {"aspect": "16:9"})
    assert meta.aspect == "16:9"
    assert not any("Aspect" in q for q in asked)


# --------------------------------------------------------------------------- #
# Casting
# --------------------------------------------------------------------------- #


POOL = [
    SystemVoice("Samantha", "en_US", "female", "adult"),
    SystemVoice("Kathy", "en_US", "female", "adult"),
    SystemVoice("Fred", "en_US", "male", "adult"),
    SystemVoice("Daniel", "en_GB", "male", "adult"),
    SystemVoice("Junior", "en_US", "male", "child"),
    SystemVoice("Grandma", "en_US", "female", "elder"),
    SystemVoice("Zarvox", "en_US", "neutral", "adult", novelty=True),
]


def test_traits_come_from_description():
    assert infer_traits(Character("X", description="A sharp woman. She runs the shift."))["gender"] == "female"
    assert infer_traits(Character("Y", description="An older man, cautious."))["gender"] == "male"
    assert infer_traits(Character("Z", description="An elderly woman."))["age"] == "elder"
    assert infer_traits(Character("W", description="A curious boy."))["age"] == "child"


def test_young_woman_is_not_a_child():
    # Regression: "young" used to match the child hint and cast leads as kids.
    assert infer_traits(Character("Maya", description="A sharp young woman."))["age"] == "adult"


def test_name_supplies_gender_when_a_script_says_nothing_else():
    """Regression: a bare `MAYA:` used to be cast with a male voice."""
    assert cast_voice(Character("Maya"), POOL).gender == "female"
    assert cast_voice(Character("Jonas"), POOL).gender == "male"


def test_description_beats_the_name_table():
    voice = cast_voice(Character("Maya", description="He is a tall man, gruff."), POOL)
    assert voice.gender == "male"


def test_novelty_voices_are_not_cast_normally():
    assert cast_voice(Character("Ada", description="A woman."), POOL).voice != "Zarvox"


def test_casting_is_deterministic_and_distinct():
    cast_a = [Character("Maya", description="a woman"), Character("Jonas", description="a man"),
              Character("Tim", description="a young boy"), Character("Ada", description="an elderly woman")]
    cast_b = [Character(c.name, description=c.description) for c in cast_a]
    cast_production(cast_a, POOL)
    cast_production(cast_b, POOL)
    assert [c.voice.voice for c in cast_a] == [c.voice.voice for c in cast_b]
    assert len({c.voice.voice for c in cast_a}) == len(cast_a), "voices must be distinct"
    assert cast_a[2].voice.voice == "Junior" and cast_a[3].voice.voice == "Grandma"


def test_casting_order_does_not_change_the_result():
    forward = [Character("Maya", description="a woman"), Character("Jonas", description="a man")]
    backward = [Character("Jonas", description="a man"), Character("Maya", description="a woman")]
    cast_production(forward, POOL)
    cast_production(backward, POOL)
    assert {c.name: c.voice.voice for c in forward} == {c.name: c.voice.voice for c in backward}


def test_existing_voices_are_never_overridden():
    member = Character("Maya", voice=VoiceProfile("system", "Daniel"))
    cast_production([member], POOL)
    assert member.voice.voice == "Daniel"


def test_shaping_stays_in_a_natural_range():
    for name in ("Maya", "Jonas", "Ada", "Tim", "Priya"):
        voice = cast_voice(Character(name), POOL)
        assert 0.85 <= voice.rate <= 1.15
        assert -5.0 <= voice.pitch <= 5.0


def test_empty_pool_degrades_to_silence():
    assert cast_voice(Character("Maya"), []).engine == "silent"


# --------------------------------------------------------------------------- #
# Speech
# --------------------------------------------------------------------------- #


def test_pitch_shift_is_accurate_in_both_directions():
    rate = 22050
    tone = np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    for semitones in (-5, -2, 2, 5):
        shifted = apply_pitch(tone, semitones)
        peak = np.fft.rfftfreq(len(shifted), 1 / rate)[np.argmax(np.abs(np.fft.rfft(shifted)))]
        assert peak == pytest.approx(220 * 2 ** (semitones / 12), rel=0.02)


def test_zero_shift_is_a_no_op():
    tone = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)
    assert np.array_equal(apply_pitch(tone, 0.0), tone)
    assert np.array_equal(apply_rate(tone, 1.0), tone)


def test_rate_change_shortens_audio():
    tone = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)
    assert len(apply_rate(tone, 2.0)) == pytest.approx(500, abs=2)


def test_normalize_scales_to_peak_and_leaves_silence_alone():
    samples = np.array([0.1, -0.05, 0.02], dtype=np.float32)
    assert float(np.max(np.abs(normalize(samples)))) == pytest.approx(0.89, abs=1e-5)
    silence = np.zeros(16, dtype=np.float32)
    assert np.array_equal(normalize(silence), silence)


def test_speech_reports_a_real_measured_duration(tmp_path):
    speech = SpeechSynthesizer(allow_neural=False)
    long_path, long_seconds = speech.speak(
        "The signal is back, and it is getting stronger every hour.",
        VoiceProfile("system", ""), tmp_path / "long.wav",
    )
    _, short_seconds = speech.speak("Yes.", VoiceProfile("system", ""), tmp_path / "short.wav")
    assert long_path.exists() and long_path.stat().st_size > 0
    assert long_seconds > short_seconds > 0


def test_voiced_and_silent_lines_are_counted(tmp_path):
    """A mute film must be detectable, not shipped quietly."""
    speech = SpeechSynthesizer(allow_neural=False)
    speech.speak("Hello there friend.", VoiceProfile("system", ""), tmp_path / "a.wav")
    assert speech.voiced_lines >= 1 or speech.silent_lines >= 1
    if speech.command:
        assert speech.voiced_lines == 1 and speech.silent_lines == 0


def test_missing_synthesiser_is_detectable(tmp_path):
    """Regression: TTS failure produced captions over silence with no warning."""
    speech = SpeechSynthesizer(allow_neural=False)
    speech.command = None
    assert not speech.available
    ok, detail = speech.probe(tmp_path / "probe.wav")
    assert not ok and detail

    before = speech.silent_lines  # probe() speaks too, so measure the delta
    speech.speak("This should have been audible.", VoiceProfile(), tmp_path / "b.wav")
    assert speech.silent_lines == before + 1 and speech.voiced_lines == 0
    assert speech.failures, "a silent fallback must record why"


def test_probe_confirms_audible_output(tmp_path):
    speech = SpeechSynthesizer(allow_neural=False)
    if speech.command:  # only meaningful where a synthesiser exists
        ok, detail = speech.probe(tmp_path / "probe.wav")
        assert ok, detail


def test_stale_silent_profile_still_speaks(tmp_path):
    """A `silent` engine saved by another machine must not mute a character."""
    speech = SpeechSynthesizer(allow_neural=False)
    if not speech.command:
        pytest.skip("no system synthesiser here")
    _, duration = speech.speak(
        "Speak this anyway.", VoiceProfile(engine="silent"), tmp_path / "c.wav"
    )
    assert speech.voiced_lines == 1 and duration > 0.2


def test_empty_line_yields_silence_not_a_crash(tmp_path):
    path, seconds = SpeechSynthesizer(allow_neural=False).speak(
        "", VoiceProfile(), tmp_path / "empty.wav"
    )
    assert path.exists() and seconds > 0


# --------------------------------------------------------------------------- #
# Music and mix
# --------------------------------------------------------------------------- #


def test_score_matches_requested_length():
    assert len(score_timeline([(0, 6, "tense"), (6, 12, "warm")], 12.0)) == int(12 * 44100)


def test_moods_produce_different_cues():
    assert not np.allclose(render_bed(3, "tense", 1), render_bed(3, "warm", 1))


@pytest.mark.parametrize("mood", sorted(MOODS))
def test_every_mood_renders_finite_audio(mood):
    bed = render_bed(2.0, mood, seed=3)
    assert np.isfinite(bed).all() and np.abs(bed).max() <= 1.2


def test_ducking_drops_the_bed_only_under_speech():
    bed = np.ones(44100 * 12, dtype=np.float32)
    ducked = duck(bed, [(1.0, 3.0), (7.0, 9.0)], depth_db=-11.0)

    def rms(signal, start, end):
        return float(np.sqrt((signal[int(start * 44100) : int(end * 44100)] ** 2).mean()))

    assert rms(ducked, 1.4, 2.6) == pytest.approx(10 ** (-11 / 20), abs=0.03)
    assert rms(ducked, 4.5, 5.5) == pytest.approx(1.0, abs=0.02)
    assert rms(ducked, 7.4, 8.6) == pytest.approx(10 ** (-11 / 20), abs=0.03)


def test_ducking_with_no_speech_is_a_no_op():
    bed = np.ones(1000, dtype=np.float32)
    assert np.array_equal(duck(bed, []), bed)


def test_mood_is_read_from_the_scene():
    assert mood_for("They ran for the door, an emergency") == "urgent"
    assert mood_for("A quiet home, family and comfort") == "warm"
    assert mood_for("nothing in particular") == "neutral"


# --------------------------------------------------------------------------- #
# Visual backends
# --------------------------------------------------------------------------- #


def request_for(**kwargs) -> ShotRequest:
    defaults = dict(prompt="a control room at night", frame_size=(320, 180), seed=42,
                    image_size=128, shot_type="medium", duration=2.0, fps=12)
    defaults.update(kwargs)
    return ShotRequest(**defaults)


def test_procedural_backend_is_always_available():
    backend = ProceduralBackend()
    assert backend.available() and STILL in backend.capabilities()


def test_procedural_plate_matches_the_frame_aspect():
    """Regression: a square plate got cropped, slicing the staged figures off."""
    visual = ProceduralBackend().render(request_for(frame_size=(320, 180)))
    width, height = visual.image.size
    assert width / height == pytest.approx(320 / 180, rel=0.02)

    portrait = ProceduralBackend().render(request_for(frame_size=(180, 320)))
    assert portrait.image.size[1] > portrait.image.size[0]


def test_procedural_output_is_deterministic_and_never_blank():
    first = ProceduralBackend().render(request_for(seed=7)).image
    second = ProceduralBackend().render(request_for(seed=7)).image
    third = ProceduralBackend().render(request_for(seed=8)).image
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(third))
    assert np.asarray(first.convert("L")).std() > 2.0  # not a flat/black frame


def test_palette_seed_holds_the_look_across_shots():
    """Regression: shots picked unrelated hues and the cuts read as glitches."""
    def mean_hue(seed):
        visual = ProceduralBackend().render(request_for(seed=seed, palette_seed=99))
        return np.asarray(visual.image.convert("RGB"), dtype=float).mean(axis=(0, 1))

    a, b = mean_hue(1), mean_hue(2)
    assert np.abs(a - b).max() < 60, "same palette seed must keep the same colour family"


def test_character_palette_is_stable_and_distinct():
    assert character_palette("Maya") == character_palette("Maya")
    assert character_palette("Maya") != character_palette("Jonas")


def test_staging_changes_with_shot_type():
    wide = ProceduralBackend().render(
        request_for(shot_type="wide", characters=[CharacterRef("Maya", speaking=True)])
    ).image
    close = ProceduralBackend().render(
        request_for(shot_type="close", characters=[CharacterRef("Maya", speaking=True)])
    ).image
    assert not np.array_equal(np.asarray(wide), np.asarray(close))


def test_speaking_character_is_rendered_differently():
    speaking = ProceduralBackend().render(
        request_for(characters=[CharacterRef("Maya", speaking=True)])
    ).image
    silent = ProceduralBackend().render(
        request_for(characters=[CharacterRef("Maya", speaking=False)])
    ).image
    assert not np.array_equal(np.asarray(speaking), np.asarray(silent))


def test_mock_backend_produces_motion_when_asked():
    visual = MockBackend().render(request_for(want_motion=True, duration=2.0, fps=12))
    assert visual.is_motion and len(visual.frames) == 24
    assert MOTION in visual.used


def test_mock_backend_degrades_to_a_still_without_motion():
    """The capability-degrade path: no motion model still yields a film."""
    visual = MockBackend(capabilities={STILL}).render(request_for(want_motion=True))
    assert not visual.is_motion and visual.image is not None
    assert MOTION not in visual.used


def test_identity_is_used_only_when_a_reference_exists(tmp_path):
    reference = tmp_path / "face.png"
    Image.new("RGB", (8, 8)).save(reference)
    with_ref = MockBackend().render(
        request_for(characters=[CharacterRef("Maya", reference_image=reference)])
    )
    without = MockBackend().render(request_for(characters=[CharacterRef("Maya")]))
    assert IDENTITY in with_ref.used
    assert IDENTITY not in without.used


def test_shot_visual_must_carry_something():
    with pytest.raises(ValueError):
        ShotVisual(backend="x")


def test_registry_selection_and_reporting():
    registry = BackendRegistry([MockBackend(), ProceduralBackend()])
    assert registry.select().name == "procedural"  # priority 10 beats mock's 5
    assert registry.select("mock").name == "mock"
    with pytest.raises(ValueError, match="not available"):
        registry.select("nonexistent")
    names = {row["name"] for row in registry.report()}
    assert names == {"mock", "procedural"}


def test_registry_skips_unavailable_backends():
    registry = BackendRegistry([MockBackend(is_available=False), ProceduralBackend()])
    assert [b.name for b in registry.available()] == ["procedural"]
    report = {row["name"]: row for row in registry.report()}
    assert report["mock"]["available"] is False
    assert report["mock"]["capabilities"] == []


def test_registry_survives_a_backend_that_raises_on_probe():
    class Exploding:
        name, priority = "boom", 999

        def available(self):
            raise RuntimeError("probe failed")

        def capabilities(self):
            return set()

        def render(self, request):
            raise RuntimeError

        def describe(self):
            return "always explodes"

    registry = BackendRegistry([Exploding(), ProceduralBackend()])
    assert registry.select().name == "procedural"
    assert {r["name"] for r in registry.report()} == {"boom", "procedural"}


def test_diffusion_backend_reports_unavailable_here_with_a_reason():
    """It must never claim capabilities it cannot deliver on this machine."""
    from video_gen.visuals.diffusion import DiffusionBackend

    backend = DiffusionBackend()
    if not backend.available():  # the expected case on this machine
        assert backend.capabilities() == set()
        assert "torch" in backend.describe()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_shot_starts_account_for_crossfade_overlap():
    durations = [5.0, 4.0, 3.0]
    fade = FilmRenderer.crossfade_for(durations)
    assert fade == CROSSFADE_SECONDS
    starts = FilmRenderer.shot_starts(durations, fade)
    # Regression: audio placed at full shot lengths drifts by one fade per cut.
    assert starts == pytest.approx([0.0, 4.65, 8.3])
    assert starts[-1] + durations[-1] == pytest.approx(sum(durations) - 2 * fade)


def test_crossfade_is_capped_and_disabled_sensibly():
    assert 0 < FilmRenderer.crossfade_for([0.6, 0.6]) < CROSSFADE_SECONDS
    assert FilmRenderer.crossfade_for([5.0]) == 0.0
    assert FilmRenderer.crossfade_for([0.1, 0.1]) == 0.0


def test_timeline_blends_between_shots(renderer):
    from video_gen.render import RenderedShot

    red = np.zeros((180, 320, 3), np.uint8)
    red[..., 0] = 255
    blue = np.zeros((180, 320, 3), np.uint8)
    blue[..., 2] = 255
    shots = [RenderedShot(4.0, lambda t: red), RenderedShot(4.0, lambda t: blue)]
    fade = FilmRenderer.crossfade_for([4.0, 4.0])
    timeline = renderer.build_timeline(shots, fade)

    assert timeline.duration == pytest.approx(8.0 - fade)
    assert timeline.frame_function(0.5)[0, 0, 0] > 200
    assert timeline.frame_function(timeline.duration - 0.5)[0, 0, 2] > 200
    middle = timeline.frame_function(FilmRenderer.shot_starts([4.0, 4.0], fade)[1] + fade / 2)
    assert middle[0, 0, 0] > 20 and middle[0, 0, 2] > 20  # both shots contribute


def test_timeline_covers_its_whole_duration(renderer):
    from video_gen.render import RenderedShot

    frame = np.zeros((180, 320, 3), np.uint8)
    durations = [3.0, 2.5, 4.0]
    shots = [RenderedShot(d, lambda t: frame) for d in durations]
    timeline = renderer.build_timeline(shots, FilmRenderer.crossfade_for(durations))
    for i in range(41):
        assert timeline.frame_function(timeline.duration * i / 40).shape == (180, 320, 3)


def test_camera_moves_actually_move(renderer):
    image = ProceduralBackend().render(request_for(image_size=128)).image
    for camera in ("push in", "pan left", "static"):
        frame_at = renderer._base_renderer(
            ShotVisual("x", image=image), duration=3.0, camera=camera, seed=1
        )
        assert not np.array_equal(frame_at(0.0), frame_at(3.0)), f"{camera} produced a freeze"


def test_motion_frames_are_played_back_and_held(renderer):
    frames = [Image.new("RGB", (64, 36), (i * 20, 0, 0)) for i in range(5)]
    frame_at = renderer._base_renderer(
        ShotVisual("x", frames=frames, frame_rate=5.0), duration=4.0, camera="static", seed=1
    )
    assert frame_at(0.0).shape == (180, 320, 3)
    assert not np.array_equal(frame_at(0.0), frame_at(0.6))
    # Past the end of a short clip the last frame is held, never an index error.
    assert np.array_equal(frame_at(3.9), frame_at(0.99))


def test_caption_carries_a_speaker_label_and_stays_in_frame(renderer):
    caption = renderer.build_caption(Line("Maya", "We are out of time."), (255, 0, 0))
    assert caption is not None
    x0, y0, box_w, box_h = caption.box
    width, height = renderer.config.resolution
    assert 0 <= x0 and x0 + box_w <= width
    assert 0 <= y0 and y0 + box_h <= height
    assert y0 > height / 2, "captions belong in the lower third"


def test_narration_captions_have_no_speaker_label(renderer):
    labelled = renderer.build_caption(Line("Maya", "Hello there friend."), (255, 0, 0))
    narrated = renderer.build_caption(Line(NARRATOR, "Hello there friend."), (255, 0, 0))
    assert narrated.box[3] < labelled.box[3]  # no label row


def test_empty_caption_is_none(renderer):
    assert renderer.build_caption(Line("Maya", "   "), (255, 0, 0)) is None


def test_long_caption_is_clamped_to_the_frame(renderer):
    caption = renderer.build_caption(Line("Maya", " ".join(["verylongword"] * 200)), (0, 255, 0))
    assert caption.box[3] <= renderer.config.resolution[1]


def test_shot_image_cache_survives_cleanup(config):
    """Regression: the cache used to live in temp_dir and be wiped every run."""
    registry = BackendRegistry([ProceduralBackend(config)])
    production = sample_production()
    with FilmRenderer(config, registry=registry) as film:
        film.shot_visual(production, production.shots[0], 3.0)
        temp_dir = film.temp_dir
        assert list(film.image_cache_dir.glob("*.png"))
    assert not temp_dir.exists()
    assert list((config.cache_dir / "images").glob("*.png"))


def test_cleanup_is_idempotent_and_recursive(config):
    film = FilmRenderer(config, registry=BackendRegistry([ProceduralBackend(config)]))
    (film.temp_dir / "nested").mkdir()
    (film.temp_dir / "nested" / "a.wav").write_bytes(b"x")
    film.cleanup()
    assert not film.temp_dir.exists()
    film.cleanup()  # must not raise


def test_renderer_falls_back_when_a_backend_raises(config):
    class Broken:
        name, priority = "broken", 500

        def available(self):
            return True

        def capabilities(self):
            return {STILL}

        def render(self, request):
            raise RuntimeError("no model")

        def describe(self):
            return "always fails"

    with FilmRenderer(config, registry=BackendRegistry([Broken()])) as film:
        production = sample_production()
        visual = film.shot_visual(production, production.shots[0], 3.0)
    assert visual.image is not None  # degraded to procedural rather than dying


# --------------------------------------------------------------------------- #
# The writers' room
# --------------------------------------------------------------------------- #


def mock_room(*replies):
    """A writers' room wired to a scripted model, so no LLM need be installed."""
    return WritersRoom(client=MockLLM(replies=list(replies)), seed=1)


def dialogue_reply(*pairs):
    return json.dumps({"lines": [{"character": c, "text": t} for c, t in pairs]})


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AMA: Morning.", "Morning."),                    # speaker prefix re-emitted
        ("Ama (warmly): Morning, love.", "Morning, love."),
        ("(whispering) They can hear us.", "They can hear us."),
        ('"The first cup is free."', "The first cup is free."),
        ("we open at six", "We open at six."),            # cased and punctuated
        ("Sit down (he gestures)", "Sit down."),          # trailing direction removed
        ("**Bold** claim here", "Bold claim here."),
    ],
)
def test_clean_line_reduces_a_model_reply_to_speech(raw, expected):
    cleaned = clean_line(raw)
    assert cleaned is not None
    assert cleaned[0] == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "Go",                                   # a single word is not a line
        "INT. KITCHEN - DAY",
        "CUT TO: the street outside",
        "FADE IN on a dark room",
        "# Scene one",
        "- a bullet point",
        "THE END OF EVERYTHING WE KNEW",         # all caps: a card, not speech
    ],
)
def test_clean_line_rejects_what_is_not_speech(raw):
    assert clean_line(raw) is None


def test_clean_line_truncates_on_a_sentence_boundary():
    long = "We roast every morning. " * 12
    text, _ = clean_line(long)
    assert len(text.split()) <= MAX_LINE_WORDS
    assert text.endswith(".")
    # It cut between sentences rather than mid-phrase.
    assert not text.endswith("every.")


def test_clean_line_keeps_a_parenthetical_as_delivery():
    text, delivery = clean_line("Ama (whispering): Not here.")
    assert text == "Not here."
    assert delivery == "whispering"


def test_echoes_catches_the_brief_being_read_back():
    brief = "a 20 second vertical advert for Kestrel Coffee, a small independent roastery"
    assert echoes("A twenty second advert for Kestrel Coffee, an independent roastery.", brief)
    assert not echoes("The first cup is on the house.", brief)
    # Too short to judge: rejecting these would throw away good punchy lines.
    assert not echoes("Kestrel Coffee.", brief)


def test_resolve_speaker_matches_loosely_but_never_invents():
    cast = [Character(name="Ama"), Character(name="Dev Patel")]
    assert resolve_speaker("ama", cast) == "Ama"
    assert resolve_speaker("AMA", cast) == "Ama"
    assert resolve_speaker("Dev", cast) == "Dev Patel"      # first name only
    assert resolve_speaker("Narrator", cast) == NARRATOR
    assert resolve_speaker("Barista", cast) is None         # not in the cast
    assert resolve_speaker("", cast) is None


def test_validate_lines_keeps_the_good_and_reports_the_rest():
    cast = [Character(name="Ama"), Character(name="Dev")]
    rejected = []
    lines = validate_lines(
        [
            {"character": "Ama", "text": "Morning. The usual?"},
            {"character": "Ghost", "text": "I am not in this film."},
            {"character": "Dev", "text": "INT. CAFE - DAY"},
            {"character": "Dev", "text": "Please."},
            {"character": "Dev", "text": "please"},          # duplicate
            "not even an object",
        ],
        cast,
        brief="a coffee advert",
        rejected=rejected,
    )
    assert [l.text for l in lines] == ["Morning. The usual?", "Please."]
    assert [l.character for l in lines] == ["Ama", "Dev"]
    assert len(rejected) == 4


def test_validate_cast_sanitises_invented_characters():
    cast = validate_cast(
        [
            {"name": "  luna  ", "description": "a warm teacher in her forties"},
            {"name": "Luna", "description": "duplicate"},
            {"name": "NARRATOR", "description": "not a character"},
            {"name": "K41r0s!!", "description": "a graduate student, his late twenties"},
            {"name": "", "description": "nameless"},
            {"name": "Third", "description": "over the limit"},
        ],
        limit=2,
    )
    assert [c.name for c in cast] == ["Luna", "Krs"]  # digits stripped, not transliterated
    # The description survives because `infer_traits` reads it to pick a voice.
    assert "forties" in cast[0].description


def test_an_invented_cast_states_its_genders_so_the_voices_differ():
    """Both roles in a two-hander used to get the same gender of voice: the
    model's descriptions carry no cue and invented names miss the name table."""
    cast = validate_cast(
        [
            {"name": "Lena", "gender": "female", "description": "A bright barista."},
            {"name": "Jasper", "gender": "male", "description": "The owner, with a kind smile."},
        ]
    )
    assert infer_traits(cast[0])["gender"] == "female"
    assert infer_traits(cast[1])["gender"] == "male"

    voices = [v.voice for v in (cast_voice(c, POOL, taken=[]) for c in cast)]
    assert len(set(voices)) == 2

    # An already-gendered description is left alone rather than prefixed twice.
    stated = validate_cast([{"name": "Ama", "gender": "female", "description": "A woman of forty."}])
    assert stated[0].description == "A woman of forty."


def test_the_room_writes_dialogue_for_an_invented_cast():
    room = mock_room(
        json.dumps({"characters": [{"name": "Ama", "description": "a warm barista, thirties"},
                                   {"name": "Dev", "description": "a tired commuter, forties"}]}),
        dialogue_reply(("Ama", "Morning. The usual?"),
                       ("Dev", "Make it a double today."),
                       ("Ama", "First one is on the house."),
                       ("Dev", "Then I am definitely coming back.")),
        json.dumps({"setting": "A small roastery", "time_of_day": "day"}),
    )
    draft = room.develop("a 20 second advert for a coffee roastery", StoryMeta(format="ad"), [])
    assert draft is not None
    assert [c.name for c in draft.cast] == ["Ama", "Dev"]
    assert len(draft.lines) == 4
    assert draft.speakers == ["Ama", "Dev"]
    assert draft.setting == "A Small Roastery"


def test_the_room_declines_rather_than_returning_half_a_draft():
    """Too few usable lines must mean "keep what you had", not a thin film."""
    room = mock_room(
        json.dumps({"characters": [{"name": "Ama", "description": "a barista"}]}),
        dialogue_reply(("Ama", "Morning."), ("Ghost", "Who am I?"), ("Ama", "INT. CAFE - DAY")),
        json.dumps({"setting": "A cafe", "time_of_day": "day"}),
    )
    assert room.develop("an advert for coffee", StoryMeta(format="ad"), []) is None


def test_the_room_survives_a_model_that_returns_junk():
    room = WritersRoom(client=MockLLM(replies=["not json at all", "{{{", ""]), seed=1)
    assert room.develop("an advert for coffee", StoryMeta(format="ad"), []) is None
    assert room.last_error


def test_the_room_is_absent_without_a_model():
    room = WritersRoom(client=MockLLM(replies=[], is_available=False))
    assert not room.available()
    assert room.develop("anything", StoryMeta(), []) is None


def test_a_long_setting_is_a_hallucination_and_is_dropped():
    """Asked to name a place, a small model will name a real university."""
    room = mock_room(json.dumps(
        {"setting": "Chapel Hill Campus, University of North Carolina at Chapel Hill",
         "time_of_day": "day"}))
    assert room.find_setting("a film about plants") == ("", "")


def test_line_budget_follows_the_format_not_a_constant():
    ad = WritersRoom.line_budget(StoryMeta(format="ad", target_seconds=30))
    educational = WritersRoom.line_budget(StoryMeta(format="educational", target_seconds=30))
    # An advert cuts faster, so the same 30s holds more lines.
    assert ad > educational
    assert 3 <= educational <= ad <= 12


# --------------------------------------------------------------------------- #
# The screenwriter's use of the room
# --------------------------------------------------------------------------- #


AD_BRIEF = "A 20 second advert for Kestrel Coffee, a small independent roastery."


def brief_writer(*replies):
    return Screenwriter(llm_client=MockLLM(replies=list(replies)))


def test_a_brief_gains_real_dialogue_instead_of_the_narrator_reading_it_back():
    """The gap this whole phase exists to close."""
    plain = Screenwriter().write(AD_BRIEF, StoryMeta(format="ad"))
    assert all(l.is_narration for l in plain.shots[0].lines), "baseline: narrator only"

    written = brief_writer(
        json.dumps({"characters": [{"name": "Ama", "description": "a warm barista"},
                                   {"name": "Dev", "description": "a tired commuter"}]}),
        dialogue_reply(("Ama", "Morning. The usual?"),
                       ("Dev", "Make it a double today."),
                       ("Ama", "First one is on the house."),
                       ("Dev", "Then I am definitely coming back.")),
        json.dumps({"setting": "A small roastery", "time_of_day": "day"}),
    ).write(AD_BRIEF, StoryMeta(format="ad"))

    lines = [l for shot in written.shots for l in shot.lines]
    assert not any(l.is_narration for l in lines), "the narrator no longer reads the brief"
    assert {l.character for l in lines} == {"Ama", "Dev"}
    assert {c.name for c in written.cast} >= {"Ama", "Dev"}
    # One speaker per shot still holds, so the camera can cut between them.
    assert all(len(set(l.character for l in shot.lines)) <= 1 for shot in written.shots)
    assert written.config["written_by"] == "mock"


def test_a_story_is_not_rewritten_into_a_model_s_version():
    """Above the brief threshold the narration *is* the film; replacing it
    would discard the user's work rather than finish it."""
    story = " ".join(["The lighthouse keeper climbed the stairs every evening at dusk."] * 8)
    assert len(story.split()) > BRIEF_MAX_WORDS
    written = brief_writer(dialogue_reply(("Ama", "Nothing to do with this story."))).write(
        story, StoryMeta()
    )
    spoken = [l.text for shot in written.shots for l in shot.lines]
    assert all("Nothing to do" not in t for t in spoken)
    assert any("lighthouse" in t.lower() for t in spoken)


def test_dialogue_the_user_wrote_is_never_replaced():
    written = brief_writer(dialogue_reply(("Ama", "An invented line."))).write(
        SCRIPT, StoryMeta(format="short")
    )
    spoken = [l.text for shot in written.shots for l in shot.lines]
    assert "An invented line." not in spoken
    assert any("signal is back" in t for t in spoken)


def test_a_rejected_draft_leaves_the_deterministic_script_standing():
    written = brief_writer(
        json.dumps({"characters": [{"name": "Ama", "description": "a barista"}]}),
        dialogue_reply(("Ghost", "Not in the cast.")),
    ).write(AD_BRIEF, StoryMeta(format="ad"))
    assert written.shots, "still a filmable screenplay"
    assert "written_by" not in written.config


def test_an_opening_imperative_is_not_a_character():
    """"Explain photosynthesis..." cast a character called Explain, which then
    stopped the room inventing anyone real, because the film had a "cast"."""
    production = Screenwriter().write(
        "Explain photosynthesis to primary school kids", StoryMeta(format="educational")
    )
    assert "EXPLAIN" not in {c.key for c in production.cast}
    for verb in ("Show", "Describe", "Create", "Sell"):
        cast = Screenwriter().extract_characters(f"{verb} a bicycle to a new rider.")
        assert verb.upper() not in {c.key for c in cast}


def test_the_same_prompt_writes_the_same_script():
    """Every other stage is deterministic; a re-render must not rewrite the film."""
    seeds = []
    for _ in range(2):
        model = MockLLM(replies=[dialogue_reply(("A", "x"))])
        Screenwriter(llm_client=model).write(AD_BRIEF, StoryMeta(format="ad"))
        seeds.append([call["seed"] for call in model.calls])
    assert seeds[0] == seeds[1] and seeds[0]


def test_regenerating_asks_for_a_different_draft():
    first = MockLLM(replies=[dialogue_reply(("A", "x"))])
    writer = Screenwriter(llm_client=first)
    writer.write(AD_BRIEF, StoryMeta(format="ad"))
    writer.variation += 1
    writer._llm_client = second = MockLLM(replies=[dialogue_reply(("A", "x"))])
    writer._room = None
    writer.write(AD_BRIEF, StoryMeta(format="ad"))
    assert first.calls[0]["seed"] != second.calls[0]["seed"]


# --------------------------------------------------------------------------- #
# The LLM client
# --------------------------------------------------------------------------- #


def test_cloud_models_are_not_offline_and_are_never_auto_selected():
    client = OllamaClient()
    client._models = [
        OllamaModel("glm-5.1:cloud", 0),
        OllamaModel("llama3.2:1b", 1_300_000_000),
        OllamaModel("qwen3-coder:30b", 18_500_000_000),
    ]
    assert client.resolve_model() == "llama3.2:1b"

    cloud_only = OllamaClient()
    cloud_only._models = [OllamaModel("glm-5.1:cloud", 0)]
    assert cloud_only.resolve_model() == ""
    assert "not offline" in cloud_only.detail


def test_auto_selection_prefers_a_model_this_host_can_actually_run():
    client = OllamaClient()
    client._models = [
        OllamaModel("qwen3-coder:30b", 18_500_000_000),
        OllamaModel("gemma3:4b", 3_300_000_000),
    ]
    assert client.resolve_model() == "gemma3:4b"


def test_a_named_model_that_is_not_installed_is_reported_not_guessed():
    client = OllamaClient(model="does-not-exist:7b")
    client._models = [OllamaModel("llama3.2:1b", 1_300_000_000)]
    assert not client.available()
    assert "not installed" in client.detail


def test_no_server_means_no_model_rather_than_an_exception():
    client = OllamaClient(host="http://127.0.0.1:9")  # nothing listens here
    assert client.models() == []
    assert not client.available()
    with pytest.raises(LLMError):
        client.complete("s", "u")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Sure! Here you go:\n{"a": 1}\nHope that helps.', '{"a": 1}'),
        ("no json here", ""),
    ],
)
def test_extract_json_survives_a_chatty_model(raw, expected):
    assert extract_json(raw) == expected


def test_a_reply_that_is_not_json_raises_rather_than_returning_junk():
    with pytest.raises(LLMError):
        LLMReply(text="I'd be happy to help!", model="m").json()


def test_probe_fails_honestly_when_the_model_answers_nothing_useful():
    ok, detail = llm_probe(MockLLM(replies=['{"line": ""}']))
    assert not ok and "no dialogue" in detail

    ok, detail = llm_probe(MockLLM(replies=['{"line": "Morning, what can I get you?"}']))
    assert ok and "6 words" in detail


def test_probe_reports_no_model_rather_than_crashing():
    ok, detail = llm_probe(MockLLM(replies=[], is_available=False))
    assert not ok


# --------------------------------------------------------------------------- #
# Against a real model, when one happens to be installed
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_a_real_local_model_writes_usable_dialogue():
    """The only test here that touches an actual LLM. Skips without one."""
    client = discover_llm()
    if client is None:
        pytest.skip("no local LLM installed")
    room = WritersRoom(client=client, seed=11)
    draft = room.develop(
        "A 20 second advert for Kestrel Coffee, a small independent roastery.",
        StoryMeta(format="ad", target_seconds=20, tone="warm"),
        [],
    )
    assert draft is not None, f"the room produced nothing usable ({room.last_error})"
    assert len(draft.lines) >= MIN_DRAFT_LINES
    for line in draft.lines:
        assert 2 <= len(line.text.split()) <= MAX_LINE_WORDS
        assert not any(m in line.text.upper() for m in ("INT.", "EXT.", "CUT TO"))
        assert not line.text.upper().startswith(line.character.upper() + ":")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_full_film_renders_end_to_end(config):
    """The whole pipeline offline, asserting the bugs that used to break it."""
    production = Screenwriter().write(SCRIPT, StoryMeta(format="short", title="E2E"))
    production.id = "e2e"
    cast_production(production.cast, POOL)
    store = ProductionStore(config.scripts_dir)
    store.save(production)

    registry = BackendRegistry([ProceduralBackend(config)])
    with FilmRenderer(config, registry=registry) as film:
        path = film.render(production)
        temp_dir = film.temp_dir
    store.save(production)

    # 1. The MP4 outlives cleanup (it used to be written inside temp_dir).
    assert not temp_dir.exists()
    assert path.exists() and path.stat().st_size > 10_000
    assert path.parent == config.output_dir

    # 2. The render was recorded back into the screenplay with real numbers.
    saved = store.load("e2e")
    assert saved.status == "rendered" and len(saved.renders) == 1
    record = saved.renders[0]
    assert record["elapsed_s"] > 0 and record["shots"] == len(production.shots)
    # 2b. The render records whether the dialogue was actually audible.
    assert record["voiced_lines"] + record["silent_lines"] == sum(
        len(s.lines) for s in saved.shots
    )
    assert set(record["cast"]) == {"Maya", "Jonas"}
    assert len({record["cast"]["Maya"], record["cast"]["Jonas"]}) == 2, "voices must differ"

    # 3. Encoded length == sum(shots) - one crossfade per cut, not the raw sum.
    durations = [s.duration for s in saved.shots]
    fade = FilmRenderer.crossfade_for(durations)
    assert record["duration_s"] == pytest.approx(sum(durations) - fade * (len(durations) - 1), abs=0.4)


@pytest.mark.slow
def test_motion_backend_renders_a_film(config):
    """Exercises the frame-sequence path that no local model can drive."""
    production = Screenwriter().write(SCRIPT, StoryMeta(title="Motion"))
    production.id = "motion"
    cast_production(production.cast, POOL)
    registry = BackendRegistry([MockBackend(config)])
    with FilmRenderer(config, registry=registry) as film:
        assert film.backend.name == "mock"
        path = film.render(production)
    assert path.exists() and path.stat().st_size > 5_000
    assert production.renders[0]["backend"] == "mock"


@pytest.mark.slow
def test_portrait_film_renders(config, tmp_path):
    """9:16 has to work end to end, not just in the config."""
    vertical = GenerationConfig(
        preset="draft", aspect="9:16", resolution=(180, 320), fps=12,
        output_dir=tmp_path / "out", scripts_dir=tmp_path / "s",
        characters_dir=tmp_path / "c", cache_dir=tmp_path / "cache",
    )
    production = Screenwriter().write(SCRIPT, StoryMeta(title="Vertical", aspect="9:16"))
    production.id = "vertical"
    cast_production(production.cast, POOL)
    with FilmRenderer(vertical, registry=BackendRegistry([ProceduralBackend(vertical)])) as film:
        path = film.render(production)
    assert path.exists() and path.stat().st_size > 5_000


# --------------------------------------------------------------------------- #
# Hosted providers: the config layer
#
# The thing being tested is mostly the *absence* of behaviour. Nothing here may
# open a socket, and no tier may switch itself on.
# --------------------------------------------------------------------------- #


def test_no_config_file_means_every_hosted_tier_is_off(tmp_path, monkeypatch):
    """The default has to be offline, including with a stray env var unset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIDEO_GEN_PROVIDERS", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))

    config = providers.load(env={})
    assert config.source is None
    for tier in providers.TIERS:
        assert config.is_enabled(tier) is False
        assert config.spec_for(tier) is None
    assert all(row["ready"] is False for row in config.status())


def test_presets_are_listed_even_with_no_config(tmp_path, monkeypatch):
    """`--providers` has to be able to say what *could* be switched on."""
    monkeypatch.chdir(tmp_path)
    config = providers.load(env={})
    assert {"agensi", "lyria", "elevenlabs", "minimax", "elevenlabs-tts"} <= set(config.specs)
    assert config.specs["elevenlabs"].tier == providers.MUSIC
    assert config.specs["elevenlabs-tts"].tier == providers.SPEECH


def test_a_config_file_switches_one_tier_on(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(
        "music:\n  enabled: true\n  provider: elevenlabs\n"
        "speech:\n  enabled: false\n  provider: elevenlabs-tts\n",
        encoding="utf-8",
    )
    config = providers.load(path, env={"ELEVENLABS_API_KEY": "k"})
    assert config.source == path
    assert config.is_enabled(providers.MUSIC)
    spec = config.spec_for(providers.MUSIC)
    assert spec is not None and spec.name == "elevenlabs"
    # The other tier stays off: one switch must not move the other.
    assert config.spec_for(providers.SPEECH) is None


def test_a_key_comes_from_the_environment_not_the_file(tmp_path, monkeypatch):
    path = tmp_path / "providers.yaml"
    path.write_text("music:\n  enabled: true\n  provider: elevenlabs\n", encoding="utf-8")

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    spec = providers.load(path).specs["elevenlabs"]
    assert spec.key() == ""
    assert "ELEVENLABS_API_KEY" in spec.missing_key_reason()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-value")
    spec = providers.load(path).specs["elevenlabs"]
    assert spec.key() == "secret-value"
    assert spec.missing_key_reason() is None


def test_a_redacted_spec_never_carries_the_key():
    spec = providers.ProviderSpec(name="x", endpoint="https://e", api_key="hunter2")
    blob = json.dumps(spec.redacted())
    assert "hunter2" not in blob
    assert spec.redacted()["api_key_present"] is True


def test_the_environment_can_switch_a_tier_on_for_one_run(tmp_path, monkeypatch):
    """An agent must be able to do this without writing a file."""
    monkeypatch.chdir(tmp_path)
    config = providers.load(
        env={"VIDEO_GEN_MUSIC_PROVIDER": "minimax", "MINIMAX_API_KEY": "k"}
    )
    assert config.is_enabled(providers.MUSIC)
    assert config.selected[providers.MUSIC] == "minimax"
    # ...and off again, explicitly.
    off = providers.load(env={"VIDEO_GEN_MUSIC_PROVIDER": "minimax", "VIDEO_GEN_MUSIC_HOSTED": "false"})
    assert off.is_enabled(providers.MUSIC) is False


def test_a_broken_config_leaves_everything_off_rather_than_raising(tmp_path):
    """A syntax error in an optional file must not stop an offline render."""
    path = tmp_path / "providers.yaml"
    path.write_text("music:\n  enabled: [unclosed\n", encoding="utf-8")
    config = providers.load(path, env={})
    assert config.is_enabled(providers.MUSIC) is False

    path.write_text("- not a mapping\n", encoding="utf-8")
    assert providers.load(path, env={}).is_enabled(providers.MUSIC) is False


def test_a_tier_pointed_at_an_unknown_or_wrong_provider_stays_off(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text("music:\n  enabled: true\n  provider: nope\n", encoding="utf-8")
    assert providers.load(path, env={}).spec_for(providers.MUSIC) is None

    # A speech provider named in the music slot is a configuration mistake, not
    # a licence to POST a music request at a TTS endpoint.
    path.write_text("music:\n  enabled: true\n  provider: elevenlabs-tts\n", encoding="utf-8")
    config = providers.load(path, env={"ELEVENLABS_API_KEY": "k"})
    assert config.spec_for(providers.MUSIC) is None


def test_a_custom_provider_inherits_a_preset_shape(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(
        "music:\n"
        "  enabled: true\n"
        "  provider: mine\n"
        "  providers:\n"
        "    mine:\n"
        "      preset: elevenlabs\n"
        "      endpoint: https://example.test/music\n"
        "      api_key_env: MINE_KEY\n",
        encoding="utf-8",
    )
    spec = providers.load(path, env={"MINE_KEY": "k"}).spec_for(providers.MUSIC)
    assert spec is not None
    assert spec.endpoint == "https://example.test/music"
    assert spec.api_key_env == "MINE_KEY"
    # Inherited from the preset rather than restated.
    assert spec.auth_header == "xi-api-key"
    assert spec.response == "audio"


def test_init_providers_writes_a_disabled_config(tmp_path):
    path = providers.write_example(tmp_path / "providers.yaml")
    assert path.is_file()
    config = providers.load(path, env={})
    assert config.is_enabled(providers.MUSIC) is False
    assert config.is_enabled(providers.SPEECH) is False
    with pytest.raises(FileExistsError):
        providers.write_example(path)
    assert providers.write_example(path, force=True) == path


# --------------------------------------------------------------------------- #
# Request templating
# --------------------------------------------------------------------------- #


def test_fill_substitutes_through_nested_structures():
    template = {"a": "{prompt}", "b": [{"c": "take {seed}"}], "d": 7}
    out = providers.fill(template, {"prompt": "hello", "seed": 3})
    assert out == {"a": "hello", "b": [{"c": "take 3"}], "d": 7}


def test_a_lone_placeholder_keeps_its_type():
    """`"{duration_ms:int}"` sending the string "30000" is a 400 from the API."""
    out = providers.fill({"len": "{duration_ms:int}", "x": "n={seed}"}, {"duration_ms": 30000.4, "seed": 2})
    assert out["len"] == 30000 and isinstance(out["len"], int)
    assert out["x"] == "n=2"  # interpolated into text, so it stays a string


def test_an_unknown_placeholder_is_left_alone_rather_than_blanked():
    assert providers.fill("{nope}", {"a": 1}) == "{nope}"


@pytest.mark.parametrize(
    "selector,response,expected",
    [
        ("audio", b"RAWBYTES", b"RAWBYTES"),
        ("base64:d.v", {"d": {"v": "aGVsbG8="}}, b"hello"),
        ("json:predictions.0.url", {"predictions": [{"url": "https://x/y.mp3"}]}, None),
        ("json:d.a", {"d": {"a": "data:audio/mp3;base64,aGVsbG8="}}, b"hello"),
        ("json:d.a", {"d": {"a": "68656c6c6f" * 8}}, bytes.fromhex("68656c6c6f" * 8)),
    ],
)
def test_extract_finds_audio_wherever_a_vendor_put_it(selector, response, expected):
    audio, url = providers.extract(response, selector)
    assert audio == expected
    if expected is None:
        assert url == "https://x/y.mp3"


@pytest.mark.parametrize(
    "selector,response",
    [
        ("json:a.b", {"a": {}}),
        ("json:a.99", {"a": [1]}),
        ("base64:a", {"a": "not base64 !!!"}),
        ("json:a.b", b"not json at all"),
    ],
)
def test_extract_reports_nothing_rather_than_guessing(selector, response):
    assert providers.extract(response, selector) == (None, None)


# --------------------------------------------------------------------------- #
# Music backends
#
# The guarantee under test everywhere here: a score never fails a render. A
# hosted provider that breaks costs fidelity, never the film.
# --------------------------------------------------------------------------- #


def score_request(duration: float = 4.0, **kwargs) -> ScoreRequest:
    kwargs.setdefault("segments", [(0.0, duration / 2, "tense"), (duration / 2, duration, "warm")])
    kwargs.setdefault("sample_rate", 8000)
    return ScoreRequest(total_duration=duration, seed=7, **kwargs)


def test_the_procedural_score_is_always_available_and_fills_the_timeline():
    backend = ProceduralMusic()
    assert backend.available() is True
    assert backend.capabilities() == {BED, CUE}
    result = backend.render(score_request(4.0))
    assert result.backend == "procedural"
    # Within a sample of the asked-for length, and audible.
    assert abs(len(result.samples) / 8000 - 4.0) < 0.01
    assert float(np.max(np.abs(result.samples))) > 0.01


def test_the_procedural_score_is_deterministic():
    a = ProceduralMusic().render(score_request())
    b = ProceduralMusic().render(score_request())
    assert np.array_equal(a.samples, b.samples)


def test_a_mood_change_actually_changes_the_music():
    """Per-scene cues are the point; one looped pad would be a regression."""
    tense = ProceduralMusic().render(score_request(segments=[(0.0, 4.0, "tense")]))
    warm = ProceduralMusic().render(score_request(segments=[(0.0, 4.0, "warm")]))
    assert not np.allclose(tense.samples, warm.samples)


def test_the_registry_prefers_the_higher_priority_backend():
    registry = MusicRegistry([ProceduralMusic(), MockMusic()])
    assert registry.select().name == "mock"  # priority 99 beats 10
    assert registry.select("procedural").name == "procedural"


def test_an_unavailable_backend_is_not_selected():
    class Off(ProceduralMusic):
        name = "off"
        priority = 500

        def available(self) -> bool:
            return False

    registry = MusicRegistry([Off(), ProceduralMusic()])
    assert registry.select().name == "procedural"
    with pytest.raises(ValueError):
        registry.select("off")


def test_a_failing_hosted_backend_falls_back_instead_of_failing_the_render():
    """The whole reason the registry takes a chain rather than one backend."""
    broken = MockMusic(fail=True)
    registry = MusicRegistry([broken, ProceduralMusic()])
    result = registry.render(score_request())
    assert broken.calls == 1  # it really was tried
    assert result.backend == "procedural"
    assert result.samples.size


def test_a_backend_returning_silence_also_falls_back():
    """An empty array is a failure too, and a subtler one than an exception."""
    registry = MusicRegistry([MockMusic(empty=True), ProceduralMusic()])
    assert registry.render(score_request()).backend == "procedural"


def test_every_backend_failing_raises_with_all_the_reasons():
    registry = MusicRegistry([MockMusic(fail=True)])
    with pytest.raises(RuntimeError) as caught:
        registry.render(score_request())
    assert "mock" in str(caught.value)


def test_a_broken_availability_probe_does_not_take_the_registry_down():
    class Exploding(ProceduralMusic):
        name = "exploding"
        priority = 900

        def available(self) -> bool:
            raise RuntimeError("probe blew up")

    registry = MusicRegistry([Exploding(), ProceduralMusic()])
    assert registry.select().name == "procedural"
    rows = {row["name"]: row for row in registry.report()}
    assert rows["exploding"]["available"] is False


def test_the_report_lists_every_backend_available_or_not():
    rows = MusicRegistry([MockMusic(), ProceduralMusic()]).report()
    assert [row["name"] for row in rows] == ["mock", "procedural"]
    assert all(set(row) >= {"name", "available", "capabilities", "detail"} for row in rows)


# -- fitting a hosted cue to the timeline ----------------------------------- #


def test_a_short_hosted_cue_is_looped_to_fill_the_timeline():
    """Providers return whatever length they like; a silent tail is not an option."""
    result = ScoreResult("x", np.ones(1000, dtype=np.float32) * 0.5, 1000)
    fitted = result.fitted(5.0, 1000)
    assert len(fitted) == 5000
    assert float(np.max(np.abs(fitted))) > 0.1


def test_a_long_hosted_cue_is_trimmed_not_sped_up():
    result = ScoreResult("x", np.ones(9000, dtype=np.float32), 1000)
    assert len(result.fitted(2.0, 1000)) == 2000


def test_a_hosted_cue_at_the_wrong_rate_is_resampled():
    result = ScoreResult("x", np.ones(2000, dtype=np.float32), 2000)  # 1s at 2kHz
    assert len(result.fitted(1.0, 8000)) == 8000


def test_an_empty_result_fits_to_silence_rather_than_crashing():
    fitted = ScoreResult("x", np.zeros(0, dtype=np.float32), 1000).fitted(2.0, 1000)
    assert len(fitted) == 2000 and not fitted.any()


def test_a_stereo_result_is_folded_to_mono():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    assert ScoreResult("x", stereo, 1000).samples.shape == (100,)


def test_a_prompt_is_built_from_the_moods_for_models_that_take_one():
    request = score_request(segments=[(0.0, 2.0, "tense"), (2.0, 4.0, "warm")], title="The Signal")
    prompt = request.prompt_for()
    assert "tense" in prompt and "warm" in prompt
    assert "The Signal" in prompt
    # The two things that make a score usable under dialogue.
    assert "no vocals" in prompt and "dialogue" in prompt
    # An explicit prompt is never overwritten.
    assert score_request(prompt="exact words").prompt_for() == "exact words"


# --------------------------------------------------------------------------- #
# The hosted music backend
#
# No test here reaches the network: urlopen is replaced. What is being checked
# is the request we *would* have sent, and that every failure degrades.
# --------------------------------------------------------------------------- #


class FakeResponse:
    """Enough of http.client.HTTPResponse for urlopen's context manager."""

    def __init__(self, body: bytes, content_type: str = "audio/mpeg") -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self, amount: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def wav_bytes(tmp_path):
    """A real wav, so decode_audio is exercised by the bundled ffmpeg."""
    from video_gen.speech import write_wav

    path = tmp_path / "cue.wav"
    t = np.arange(8000, dtype=np.float32) / 8000
    write_wav(path, (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 8000)
    return path.read_bytes()


def hosted_config(tmp_path, response="audio", **extra):
    """A HostedMusic wired to a one-provider config, with caching in tmp_path."""
    spec = providers.ProviderSpec(
        name="test-music",
        tier=providers.MUSIC,
        endpoint="https://example.test/v1/music",
        api_key="key-123",
        auth_header="x-api-key",
        auth_prefix="",
        body={"prompt": "{prompt}", "ms": "{duration_ms:int}"},
        response=response,
        **extra,
    )
    config = providers.ProvidersConfig(
        enabled={providers.MUSIC: True, providers.SPEECH: False},
        selected={providers.MUSIC: "test-music", providers.SPEECH: ""},
        specs={"test-music": spec},
    )
    return config, spec


def test_hosted_music_is_unavailable_with_no_config():
    """Off by default means `available()` is False, not a connection attempt."""
    backend = HostedMusic(None, providers.ProvidersConfig())
    assert backend.available() is False
    assert "disabled" in backend.describe()
    with pytest.raises(RuntimeError):
        backend.render(score_request())


def test_hosted_music_is_unavailable_without_a_key(tmp_path):
    config, spec = hosted_config(tmp_path)
    spec.api_key = ""
    spec.api_key_env = "DEFINITELY_NOT_SET_12345"
    backend = HostedMusic(None, config)
    assert backend.available() is False
    assert "DEFINITELY_NOT_SET_12345" in backend.describe()


def test_hosted_music_sends_the_request_the_provider_documents(tmp_path, wav_bytes, monkeypatch, config):
    providers_config, _ = hosted_config(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse(wav_bytes, "audio/wav")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = HostedMusic(config, providers_config)
    assert backend.available() is True
    result = backend.render(score_request(30.0))

    assert captured["url"] == "https://example.test/v1/music"
    assert captured["method"] == "POST"
    # Header names are capitalised by urllib; the value is what matters.
    assert captured["headers"]["X-api-key"] == "key-123"
    assert captured["body"]["ms"] == 30000 and isinstance(captured["body"]["ms"], int)
    assert "tense" in captured["body"]["prompt"]
    assert result.backend == "hosted" and result.samples.size


def test_hosted_music_follows_a_url_when_that_is_what_came_back(tmp_path, wav_bytes, monkeypatch, config):
    providers_config, _ = hosted_config(tmp_path, response="json:data.url")
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if len(calls) == 1:
            return FakeResponse(
                json.dumps({"data": {"url": "https://cdn.test/a.wav"}}).encode(),
                "application/json",
            )
        return FakeResponse(wav_bytes, "audio/wav")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = HostedMusic(config, providers_config).render(score_request())
    assert calls == ["https://example.test/v1/music", "https://cdn.test/a.wav"]
    assert result.samples.size


def test_a_hosted_result_is_cached_so_a_re_render_does_not_pay_twice(tmp_path, wav_bytes, monkeypatch, config):
    """Each call costs money, so the cache matters more here than for images."""
    providers_config, _ = hosted_config(tmp_path)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        return FakeResponse(wav_bytes, "audio/wav")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    request = score_request(3.0)
    first = HostedMusic(config, providers_config).render(request)
    second = HostedMusic(config, providers_config).render(request)
    assert len(calls) == 1
    assert "cached" in second.detail
    assert np.allclose(first.samples, second.samples)


def test_a_different_request_is_not_served_from_the_cache(tmp_path, wav_bytes, monkeypatch, config):
    providers_config, _ = hosted_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: (calls.append(1), FakeResponse(wav_bytes, "audio/wav"))[1],
    )
    backend = HostedMusic(config, providers_config)
    backend.render(score_request(3.0))
    backend.render(score_request(9.0))
    assert len(calls) == 2


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b"not audio at all", "audio/mpeg"),
        (b'{"unexpected": "shape"}', "application/json"),
        (b"", "audio/mpeg"),
    ],
)
def test_an_undecodable_response_raises_so_the_registry_can_fall_back(
    tmp_path, monkeypatch, config, body, content_type
):
    providers_config, _ = hosted_config(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: FakeResponse(body, content_type),
    )
    with pytest.raises(RuntimeError):
        HostedMusic(config, providers_config).render(score_request())


def test_an_http_error_is_reported_with_the_providers_own_message(tmp_path, monkeypatch, config):
    import io
    import urllib.error

    providers_config, _ = hosted_config(tmp_path)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 402, "Payment Required", {}, io.BytesIO(b"out of credit")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as caught:
        HostedMusic(config, providers_config).render(score_request())
    assert "402" in str(caught.value) and "out of credit" in str(caught.value)


def test_a_hosted_failure_still_produces_a_film_through_the_registry(tmp_path, monkeypatch, config):
    """End to end: the provider is down and the score comes out anyway."""
    import urllib.error

    providers_config, _ = hosted_config(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("no route")),
    )
    registry = MusicRegistry([HostedMusic(config, providers_config), ProceduralMusic()])
    result = registry.render(score_request())
    assert result.backend == "procedural"
    assert result.samples.size


def test_decode_audio_round_trips_a_real_wav(wav_bytes):
    samples = decode_audio(wav_bytes, 8000)
    assert abs(len(samples) / 8000 - 1.0) < 0.05
    assert 0.1 < float(np.max(np.abs(samples))) <= 1.0


def test_decode_audio_rejects_rubbish_rather_than_returning_noise():
    assert decode_audio(b"", 8000).size == 0
    with pytest.raises(RuntimeError):
        decode_audio(b"\x00\x01\x02 not a container", 8000)


# --------------------------------------------------------------------------- #
# Linux voices
#
# espeak-ng cannot be installed on the development machine, so these tests use
# its real listing format as a fixture and a stub binary for the synthesis
# path. That stub is what proves `-v` is actually passed -- the bug that made
# every character on Linux share one voice.
# --------------------------------------------------------------------------- #

ESPEAK_EN = """\
Pty Language       Age/Gender VoiceName          File                 Other Languages
 5  en-029          --/M      en-caribbean       gmw/en-029
 2  en-gb           --/M      en-english         gmw/en
 5  en-gb-scotland  --/M      en-scottish        gmw/en-GB-scotland   (en 4)
 5  en-gb-x-gbclan  --/M      en-english-north   gmw/en-GB-x-gbclan   (en-gb 3)
 5  en-us           --/M      en-us              gmw/en-US            (en 3)
 7  af              --/M      Afrikaans          gmw/af
"""

ESPEAK_VARIANTS = """\
Pty Language       Age/Gender VoiceName          File                 Other Languages
 5  variant         --/M      Alicia             !v/Alicia
 5  variant         --/F      Annie              !v/Annie
 5  variant          -/M      m1                 !v/m1
 5  variant          -/M      m2                 !v/m2
 5  variant          -/M      m3                 !v/m3
 5  variant          -/F      f1                 !v/f1
 5  variant          -/F      f2                 !v/f2
 5  variant          -/F      f3                 !v/f3
 5  variant          -/M      croak              !v/croak
 5  variant          -/M      whisper            !v/whisper
 5  variant          -/M      robosoft           !v/robosoft
"""


def test_the_espeak_listing_format_parses():
    rows = _parse_espeak_rows(ESPEAK_EN)
    assert len(rows) == 6  # the header is not a row
    assert rows[1] == {
        "language": "en-gb", "age": "", "gender": "M",
        "name": "en-english", "file": "gmw/en",
    }
    assert _parse_espeak_rows("") == []
    assert _parse_espeak_rows("garbage\nmore garbage") == []


def fake_espeak(monkeypatch, en=ESPEAK_EN, variants=ESPEAK_VARIANTS):
    """Answer `--voices=` without an espeak binary present."""
    def fake_list(command, selector):
        from video_gen.casting import _parse_espeak_rows as parse

        return parse(en if selector != "variant" else variants)

    monkeypatch.setattr("video_gen.casting._espeak_list", fake_list)


def test_linux_gets_a_real_voice_pool_not_one_voice(monkeypatch):
    """The whole point: 'a voice per character' has to mean something here."""
    fake_espeak(monkeypatch)
    voices = discover_espeak_voices("espeak-ng", "en")
    assert len(voices) > 10
    names = [v.name for v in voices]
    assert len(set(names)) == len(names)  # no duplicates
    # Both genders, or casting cannot tell two characters apart.
    assert sum(v.gender == "female" for v in voices) >= 3
    assert sum(v.gender == "male" for v in voices) >= 3


def test_the_base_accents_come_first_and_are_plain(monkeypatch):
    fake_espeak(monkeypatch)
    voices = discover_espeak_voices("espeak-ng", "en")
    assert [v.name for v in voices[:5]] == [
        "en-029", "en-gb", "en-gb-scotland", "en-gb-x-gbclan", "en-us",
    ]
    assert {v.accent for v in voices[:5]} == {"029", "gb", "us"}


def test_only_the_asked_for_language_is_offered(monkeypatch):
    """`af` is in the listing and must not be cast in an English film."""
    fake_espeak(monkeypatch)
    voices = discover_espeak_voices("espeak-ng", "en")
    assert not any(v.name.startswith("af") for v in voices)


def test_variants_are_combined_with_the_base_accents(monkeypatch):
    fake_espeak(monkeypatch)
    names = [v.name for v in discover_espeak_voices("espeak-ng", "en")]
    combined = [n for n in names if "+" in n]
    assert combined, "no variant voices were built"
    assert all(n.split("+")[0].startswith("en") for n in combined)
    # Variants are spread across accents rather than all applied to one.
    assert len({n.split("+")[0] for n in combined}) > 1


def test_a_variants_gender_overrides_the_bases(monkeypatch):
    fake_espeak(monkeypatch)
    by_name = {v.name: v for v in discover_espeak_voices("espeak-ng", "en")}
    female = [v for name, v in by_name.items() if "+f" in name or "+Annie" in name]
    assert female and all(v.gender == "female" for v in female)


def test_novelty_variants_are_kept_out_of_the_pool(monkeypatch):
    """A croaking robot is not a performer."""
    fake_espeak(monkeypatch)
    names = [v.name for v in discover_espeak_voices("espeak-ng", "en")]
    for junk in ("croak", "whisper", "robosoft"):
        assert not any(junk in n for n in names)


def test_the_pool_is_capped_so_doctor_stays_readable(monkeypatch):
    from video_gen.casting import MAX_ESPEAK_VOICES

    many = ESPEAK_VARIANTS + "".join(
        f" 5  variant          -/M      extra{i}            !v/extra{i}\n" for i in range(200)
    )
    fake_espeak(monkeypatch, variants=many)
    assert len(discover_espeak_voices("espeak-ng", "en")) == MAX_ESPEAK_VOICES


def test_an_unparseable_listing_still_yields_one_working_voice(monkeypatch):
    """One voice beats zero: zero means a silent film."""
    fake_espeak(monkeypatch, en="totally unexpected output")
    voices = discover_espeak_voices("espeak-ng", "en")
    assert len(voices) == 1 and voices[0].name == "en"


def test_espeak_voices_cast_distinctly(monkeypatch):
    """The pool is only useful if two characters end up on different voices."""
    fake_espeak(monkeypatch)
    pool = discover_espeak_voices("espeak-ng", "en")
    cast = [
        Character("Maya", description="A sharp young woman."),
        Character("Jonas", description="An older man, cautious."),
        Character("Ada", description="A woman in her thirties."),
    ]
    cast_production(cast, pool)
    chosen = [c.voice.voice for c in cast]
    assert all(chosen) and len(set(chosen)) == 3


# -- the stub binary: proves -v reaches the synthesiser --------------------- #


@pytest.fixture
def stub_espeak(tmp_path, monkeypatch):
    """A fake espeak-ng on PATH that logs its argv and writes a real wav."""
    binary = tmp_path / "bin" / "espeak-ng"
    binary.parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "argv.log"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys, wave, math, struct\n"
        f"open({str(log)!r}, 'a').write('\\t'.join(sys.argv[1:]) + '\\n')\n"
        "argv = sys.argv[1:]\n"
        "if any(a.startswith('--voices') for a in argv):\n"
        f"    sel = [a for a in argv if a.startswith('--voices')][0].split('=')[1]\n"
        f"    sys.stdout.write({ESPEAK_VARIANTS!r} if sel == 'variant' else {ESPEAK_EN!r})\n"
        "    sys.exit(0)\n"
        "out = argv[argv.index('-w') + 1]\n"
        "with wave.open(out, 'wb') as h:\n"
        "    h.setnchannels(1); h.setsampwidth(2); h.setframerate(22050)\n"
        "    h.writeframes(b''.join(struct.pack('<h', int(12000 * math.sin(i / 8.0)))\n"
        "                           for i in range(11025)))\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    # Prepended rather than replacing PATH: the stub's own interpreter has to
    # stay reachable, and the espeak branch is forced below regardless.
    monkeypatch.setenv("PATH", f"{binary.parent}{os.pathsep}{os.environ['PATH']}")
    # system_tts_command prefers `say` on darwin, so force the espeak branch.
    monkeypatch.setattr("video_gen.casting.sys.platform", "linux")
    return log


def test_the_espeak_path_synthesises_audible_speech(stub_espeak, tmp_path):
    from video_gen.casting import system_tts_command

    assert system_tts_command() == "espeak-ng"
    speech = SpeechSynthesizer(allow_neural=False)
    assert speech.command == "espeak-ng"
    ok, detail = speech.probe(tmp_path / "probe.wav")
    assert ok, detail
    assert speech.voiced_lines == 1 and speech.silent_lines == 0


def test_the_cast_voice_is_passed_to_espeak(stub_espeak, tmp_path):
    """Without -v every character on Linux spoke in the default voice."""
    speech = SpeechSynthesizer(allow_neural=False)
    speech.speak("Hello there.", VoiceProfile(voice="en-gb+f3", rate=1.1, pitch=-2.0), tmp_path / "a.wav")
    argv = stub_espeak.read_text().strip().splitlines()[-1].split("\t")
    assert "-v" in argv
    assert argv[argv.index("-v") + 1] == "en-gb+f3"
    # Rate and pitch still reach it too.
    assert "-s" in argv and "-p" in argv


def test_two_characters_on_linux_get_two_different_voices(stub_espeak, tmp_path):
    """End to end on the espeak path: discovery, casting and synthesis."""
    from video_gen.casting import discover_voices as discover

    pool = discover()
    assert len(pool) > 5
    cast = [
        Character("Maya", description="A sharp young woman."),
        Character("Jonas", description="An older man."),
    ]
    cast_production(cast, pool)
    assert cast[0].voice.voice != cast[1].voice.voice

    speech = SpeechSynthesizer(allow_neural=False)
    for index, member in enumerate(cast):
        speech.speak("A line of dialogue.", member.voice, tmp_path / f"{index}.wav")
    assert speech.voiced_lines == 2 and speech.silent_lines == 0
    used = [
        line.split("\t")[line.split("\t").index("-v") + 1]
        for line in stub_espeak.read_text().strip().splitlines()
        if "-v" in line.split("\t")
    ]
    assert len(set(used)) == 2


# --------------------------------------------------------------------------- #
# The agent contract
#
# An agent reads the exit code, then the JSON. Both are promises, so both are
# tested as promises rather than as whatever the code happens to do.
# --------------------------------------------------------------------------- #


def test_the_exit_codes_are_distinct_and_named():
    codes = [EXIT_OK, EXIT_USAGE, EXIT_NO_AUDIO, EXIT_FAILED, EXIT_CANCELLED]
    assert codes == [0, 1, 2, 3, 4]
    assert len(set(codes)) == len(codes)
    assert all(code in agent.EXIT_MEANINGS for code in codes)


def test_every_payload_carries_the_keys_a_caller_branches_on():
    payload = CommandResult(EXIT_OK, {"output": "x.mp4"}).payload("create")
    assert payload["schema"] == agent.SCHEMA
    assert payload["command"] == "create"
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert payload["status"] == "ok"
    # Present and null rather than absent, so a caller never has to use .get().
    assert payload["error"] is None
    assert payload["output"] == "x.mp4"


def test_a_failure_payload_says_why():
    payload = agent.fail(EXIT_FAILED, "ffmpeg died", id="t1").payload("render")
    assert payload["ok"] is False
    assert payload["exit_code"] == EXIT_FAILED
    assert payload["status"] == "render failed"
    assert payload["error"] == "ffmpeg died"
    assert payload["id"] == "t1"


def test_a_payload_is_json_serialisable_and_has_no_paths_in_it():
    """json.dump with default=str would hide a Path; the contract says string."""
    production = sample_production()
    payload = agent.production_payload(production, Path("video_scripts/t1.md"))
    encoded = json.dumps(payload)  # no default= : must already be plain types
    assert "t1.md" in encoded
    assert isinstance(payload["screenplay"], str)
    assert Path(payload["screenplay"]).is_absolute()


def test_the_screenplay_payload_carries_the_whole_script():
    production = sample_production()
    payload = agent.production_payload(production)
    assert payload["shot_count"] == len(production.shots)
    assert payload["scene_count"] == len(production.scenes)
    assert [c["name"] for c in payload["cast"]] == [c.name for c in production.cast]
    lines = [l for s in payload["scenes"] for sh in s["shots"] for l in sh["lines"]]
    assert lines and all({"character", "text", "narration"} <= set(l) for l in lines)
    # Enough to re-stage a shot without opening the markdown.
    shot = payload["scenes"][0]["shots"][0]
    assert {"shot_type", "camera", "subjects", "duration_s"} <= set(shot)


def test_a_listing_row_is_a_summary_not_a_whole_script():
    payload = agent.production_summary(sample_production())
    assert "scenes" not in payload
    assert {"id", "title", "production_status", "shot_count"} <= set(payload)


def test_a_command_can_never_redefine_an_envelope_key():
    """`status` is the outcome. A production's own status needed another name."""
    payload = CommandResult(EXIT_OK, {"status": "accepted", "ok": "nope"}).payload("show")
    assert payload["status"] == "ok"
    assert payload["ok"] is True
    assert payload["data_status"] == "accepted"
    assert payload["data_ok"] == "nope"


def test_the_render_payload_reports_silence_as_prominently_as_success():
    production = sample_production()
    production.renders.append({"duration_s": 9.5, "backend": "procedural"})
    payload = agent.render_payload(
        production, output=Path("out.mp4"), elapsed_s=12.0,
        voiced_lines=0, silent_lines=4, screenplay=Path("s.md"),
    )
    assert payload["voiced_lines"] == 0
    assert payload["silent_lines"] == 4
    assert payload["duration_s"] == 9.5


def test_the_doctor_payload_answers_what_an_agent_must_decide_on():
    from video_gen.casting import discover_voices as discover

    payload = agent.doctor_payload(
        config=GenerationConfig(), voices=discover(), speech_ok=True,
        speech_detail="fine", llm_describe="none", llm_probe=None,
        visual_rows=[], music_rows=[], provider_rows=[],
        selected_visual="procedural", selected_music="procedural",
    )
    assert {"render", "speech", "write_dialogue", "hosted_music", "hosted_speech"} == set(
        payload["capabilities"]
    )
    assert payload["capabilities"]["write_dialogue"] is False  # "none"
    assert payload["version"]
    # The vocabulary an agent needs to build a valid command.
    assert set(payload["formats"]) == set(FORMATS)
    assert set(payload["aspects"]) == set(ASPECTS)
    assert set(payload["styles"]) == set(STYLE_TIERS)
    assert json.dumps(payload)


# -- the parser ------------------------------------------------------------- #


def test_the_prompt_flag_exists_and_is_not_the_prompt_file_flag():
    """Documented in --help since v1; argparse abbreviated it to --prompt-file
    and the run died on FileNotFoundError."""
    from video_gen.cli import build_parser

    args = build_parser().parse_args(["--prompt", "a 30s advert for a bike shop"])
    assert args.prompt == "a 30s advert for a bike shop"
    assert args.prompt_file is None


def test_every_prompt_form_reaches_the_same_place():
    from video_gen.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["a story"]).prompt_positional == "a story"
    assert parser.parse_args(["--prompt", "a story"]).prompt == "a story"
    assert parser.parse_args(["--prompt-file", "s.txt"]).prompt_file == Path("s.txt")


def test_json_implies_yes_so_nothing_can_block_on_a_prompt(monkeypatch):
    """An agent cannot answer `Render now? [Y/n]`."""
    from video_gen import cli

    monkeypatch.setattr(cli, "dispatch", lambda args: CommandResult(EXIT_OK, {}))
    seen = {}
    monkeypatch.setattr(cli, "_command_name", lambda args: seen.setdefault("yes", args.yes) and "x")
    cli.main(["--json", "--doctor"])
    assert seen["yes"] is True


def test_the_help_epilogue_only_advertises_flags_that_exist():
    """The --prompt bug shipped because the epilogue was never checked."""
    import re as _re

    from video_gen.cli import build_parser

    parser = build_parser()
    known = {action.option_strings[0] for action in parser._actions if action.option_strings}
    advertised = set(_re.findall(r"(?<!\w)(--[a-z][a-z-]+)", parser.epilog or ""))
    assert advertised <= known, f"documented but undefined: {sorted(advertised - known)}"


# --------------------------------------------------------------------------- #
# The CLI as an agent actually calls it
#
# Through a real subprocess, because the promise is about stdout, stderr and
# the exit code of a process -- not about what a function returns in-process.
# --------------------------------------------------------------------------- #

import subprocess


def run_cli(*argv, cwd=None, expect=None):
    """Invoke the CLI the way an agent would and parse stdout as JSON."""
    done = subprocess.run(
        [sys.executable, "-m", "video_gen", "--json", *argv],
        cwd=str(cwd or Path(__file__).parent),
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = json.loads(done.stdout)  # must be the only thing on stdout
    if expect is not None:
        assert done.returncode == expect, f"exit {done.returncode}: {done.stdout}\n{done.stderr}"
    return done, payload


@pytest.fixture
def sandbox(tmp_path):
    """A working directory with nothing of the project's own state in it."""
    for name in ("output", "video_scripts", "characters", ".cache"):
        (tmp_path / name).mkdir()
    return tmp_path


@pytest.mark.slow
def test_stdout_carries_only_json_and_the_log_goes_to_stderr(sandbox):
    """A single stray print on stdout corrupts the payload at the caller."""
    done, payload = run_cli("--doctor", cwd=sandbox, expect=EXIT_OK)
    assert payload["command"] == "doctor"
    assert done.stdout.lstrip().startswith("{")
    assert done.stdout.rstrip().endswith("}")
    # The human report really was produced -- just not on stdout.
    assert "CAPABILITY REPORT" in done.stderr
    assert "CAPABILITY REPORT" not in done.stdout


@pytest.mark.slow
def test_doctor_reports_capabilities_an_agent_can_branch_on(sandbox):
    _, payload = run_cli("--doctor", cwd=sandbox, expect=EXIT_OK)
    able = payload["capabilities"]
    assert able["render"] is True
    assert isinstance(able["speech"], bool)
    # Nothing paid may be on by default, in a fresh directory.
    assert able["hosted_music"] is False
    assert able["hosted_speech"] is False
    assert payload["voices"]["castable"] >= 0


@pytest.mark.slow
def test_an_unknown_id_is_a_usage_error_not_a_traceback(sandbox):
    done, payload = run_cli("--render", "no-such-id", cwd=sandbox, expect=EXIT_USAGE)
    assert payload["ok"] is False
    assert "no-such-id" in payload["error"]
    assert "Traceback" not in done.stdout


@pytest.mark.slow
def test_a_missing_prompt_file_is_a_usage_error(sandbox):
    _, payload = run_cli("--prompt-file", "nope.txt", cwd=sandbox, expect=EXIT_USAGE)
    assert "nope.txt" in payload["error"]


@pytest.mark.slow
def test_an_empty_prompt_is_a_usage_error(sandbox):
    _, payload = run_cli("--prompt", "   ", cwd=sandbox, expect=EXIT_USAGE)
    assert payload["error"] == "no prompt given"


@pytest.mark.slow
def test_a_bad_backend_name_is_reported_as_json(sandbox):
    _, payload = run_cli(
        "--prompt", "MAYA: Hello there.", "--backend", "nonexistent",
        "--output", str(sandbox / "x.mp4"), cwd=sandbox, expect=EXIT_FAILED,
    )
    assert payload["ok"] is False and payload["error"]


@pytest.mark.slow
def test_the_prompt_flag_runs_rather_than_crashing(sandbox):
    """Regression: `--prompt` used to die with FileNotFoundError."""
    _, payload = run_cli(
        "--prompt", "a 30s advert for a bike shop", "--no-render",
        cwd=sandbox, expect=EXIT_OK,
    )
    assert payload["command"] == "create"
    assert payload["id"] and payload["shot_count"] >= 1


@pytest.mark.slow
def test_no_render_writes_a_screenplay_and_stops(sandbox):
    _, payload = run_cli(
        "--prompt-file", str(Path(__file__).parent / "examples" / "01_screenplay.txt"),
        "--no-render", cwd=sandbox, expect=EXIT_OK,
    )
    assert payload["status"] == "ok"  # the command's outcome
    assert payload["production_status"] == "accepted"  # the production's own
    assert Path(payload["screenplay"]).is_file()
    assert not list((sandbox / "output").glob("*.mp4"))
    assert payload["title"] == "Control Room"  # not "INT"
    assert [c["name"] for c in payload["cast"]] == ["Maya", "Jonas"]


@pytest.mark.slow
def test_list_and_show_round_trip_a_saved_production(sandbox):
    _, written = run_cli(
        "--prompt-file", str(Path(__file__).parent / "examples" / "01_screenplay.txt"),
        "--no-render", cwd=sandbox, expect=EXIT_OK,
    )
    _, listing = run_cli("--list", cwd=sandbox, expect=EXIT_OK)
    ids = [row["id"] for row in listing["productions"]]
    assert written["id"] in ids

    _, shown = run_cli("--show", written["id"], cwd=sandbox, expect=EXIT_OK)
    assert shown["id"] == written["id"]
    assert shown["shot_count"] == written["shot_count"]
    spoken = [
        line["text"]
        for scene in shown["scenes"] for shot in scene["shots"] for line in shot["lines"]
    ]
    assert "The signal is back. Same rhythm as last night." in spoken


@pytest.mark.slow
def test_characters_lists_the_reusable_cast(sandbox):
    run_cli(
        "--prompt-file", str(Path(__file__).parent / "examples" / "01_screenplay.txt"),
        "--no-render", cwd=sandbox, expect=EXIT_OK,
    )
    _, payload = run_cli("--characters", cwd=sandbox, expect=EXIT_OK)
    names = [c["name"] for c in payload["characters"]]
    assert "Maya" in names and "Jonas" in names
    assert all("voice" in c for c in payload["characters"])


@pytest.mark.slow
def test_providers_reports_everything_off_in_a_fresh_directory(sandbox):
    _, payload = run_cli("--providers", cwd=sandbox, expect=EXIT_OK)
    assert payload["config_file"] is None
    assert all(tier["ready"] is False for tier in payload["tiers"])
    assert {"agensi", "lyria", "elevenlabs", "minimax"} <= {p["name"] for p in payload["presets"]}


@pytest.mark.slow
def test_a_configured_key_never_appears_in_a_payload(sandbox, monkeypatch):
    """`--providers` is printed and logged, so the value must never be in it."""
    env = dict(os.environ, ELEVENLABS_API_KEY="super-secret-sentinel")
    done = subprocess.run(
        [sys.executable, "-m", "video_gen", "--json", "--providers"],
        cwd=str(sandbox), capture_output=True, text=True, timeout=300, env=env,
    )
    assert done.returncode == EXIT_OK
    assert "super-secret-sentinel" not in done.stdout
    assert "super-secret-sentinel" not in done.stderr
    payload = json.loads(done.stdout)
    spec = next(p for p in payload["presets"] if p["name"] == "elevenlabs")
    assert spec["api_key_present"] is True  # found, and still not disclosed


@pytest.mark.slow
def test_init_providers_scaffolds_a_disabled_config(sandbox):
    _, payload = run_cli("--init-providers", cwd=sandbox, expect=EXIT_OK)
    written = Path(payload["path"])
    assert written.is_file() and payload["enabled"] is False

    _, again = run_cli("--init-providers", cwd=sandbox, expect=EXIT_USAGE)
    assert "already exists" in again["error"]

    # And the written file really does leave everything off.
    _, status = run_cli("--providers", cwd=sandbox, expect=EXIT_OK)
    assert all(tier["ready"] is False for tier in status["tiers"])


@pytest.mark.slow
def test_version_is_reported(sandbox):
    done = subprocess.run(
        [sys.executable, "-m", "video_gen", "--version"],
        cwd=str(sandbox), capture_output=True, text=True, timeout=120,
    )
    from video_gen import __version__

    assert __version__ in done.stdout + done.stderr


# --------------------------------------------------------------------------- #
# Exit code 2: a film came out and nobody can hear it
#
# The failure this guards is the nastiest one in the project. The MP4 is the
# right length, has real subtitles and plays silence, so an agent that only
# checks "did a file appear" hands over a broken video.
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_synthesiser(monkeypatch):
    """A machine with no TTS at all, as a bare Linux container would be."""
    monkeypatch.setattr("video_gen.casting.system_tts_command", lambda: None)
    monkeypatch.setattr("video_gen.speech.system_tts_command", lambda: None)


@pytest.mark.slow
def test_a_silent_render_exits_two_not_zero(config, no_synthesiser):
    from video_gen.cli import command_self_test

    result = command_self_test(config, config.output_dir / "silent.mp4")
    assert result.code == EXIT_NO_AUDIO
    assert result.ok is False
    # A real file really was produced -- that is what makes this dangerous.
    assert Path(result.data["output"]).is_file()
    assert result.data["voiced_lines"] == 0
    assert result.data["silent_lines"] > 0
    assert "no audio" in result.data["error"].lower()


@pytest.mark.slow
def test_the_silent_payload_says_so_in_the_envelope_too(config, no_synthesiser):
    from video_gen.cli import command_self_test

    payload = command_self_test(config, config.output_dir / "s.mp4").payload("self-test")
    assert payload["ok"] is False
    assert payload["exit_code"] == EXIT_NO_AUDIO
    assert payload["status"] == "rendered with no audible dialogue"
    assert payload["error"]


@pytest.mark.slow
def test_a_render_with_sound_exits_zero(config):
    from video_gen.cli import command_self_test

    result = command_self_test(config, config.output_dir / "loud.mp4")
    if not SpeechSynthesizer().available:
        pytest.skip("no synthesiser on this machine, so there is nothing to compare")
    assert result.code == EXIT_OK
    assert result.data["voiced_lines"] > 0
    assert result.data["silent_lines"] == 0
    assert result.payload("self-test")["error"] is None


@pytest.mark.slow
def test_the_renderer_counts_silence_rather_than_pretending(config, no_synthesiser):
    production = sample_production()
    with FilmRenderer(config, registry=BackendRegistry([ProceduralBackend(config)])) as film:
        film.render(production)
    record = production.renders[-1]
    assert record["voiced_lines"] == 0
    assert record["silent_lines"] > 0


# --------------------------------------------------------------------------- #
# The renderer and the music seam
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_renderer_records_which_backend_scored_the_film(config):
    production = sample_production()
    with FilmRenderer(config, registry=BackendRegistry([ProceduralBackend(config)])) as film:
        film.render(production)
        assert film.score_backend == "procedural"
    assert production.renders[-1]["score_backend"] == "procedural"


@pytest.mark.slow
def test_the_renderer_uses_the_music_registry_it_is_given(config):
    mock = MockMusic(sample_rate=11025)
    production = sample_production()
    with FilmRenderer(
        config,
        registry=BackendRegistry([ProceduralBackend(config)]),
        music_registry=MusicRegistry([mock, ProceduralMusic()]),
    ) as film:
        film.render(production)
        assert film.score_backend == "mock"
    assert mock.calls == 1
    # The request carried the real timeline, not a placeholder.
    request = mock.requests[0]
    assert request.total_duration > 0
    assert request.segments and all(len(s) == 3 for s in request.segments)
    assert request.format == production.format


@pytest.mark.slow
def test_a_failing_score_backend_still_produces_a_film(config):
    """The guarantee, end to end through the renderer."""
    production = sample_production()
    with FilmRenderer(
        config,
        registry=BackendRegistry([ProceduralBackend(config)]),
        music_registry=MusicRegistry([MockMusic(fail=True), ProceduralMusic()]),
    ) as film:
        path = film.render(production)
        assert film.score_backend == "procedural"
    assert path.is_file() and path.stat().st_size > 5_000


@pytest.mark.slow
def test_no_music_skips_the_score_entirely(tmp_path):
    """--no-music must not call a paid provider."""
    silent_config = GenerationConfig(
        preset="draft", resolution=(320, 180), fps=12, music=False,
        output_dir=tmp_path / "o", scripts_dir=tmp_path / "s",
        characters_dir=tmp_path / "c", cache_dir=tmp_path / "k",
    )
    mock = MockMusic()
    production = sample_production()
    with FilmRenderer(
        silent_config,
        registry=BackendRegistry([ProceduralBackend(silent_config)]),
        music_registry=MusicRegistry([mock]),
    ) as film:
        film.render(production)
    assert mock.calls == 0
    assert film.score_backend == ""


@pytest.mark.slow
def test_a_scene_mood_reaches_the_score_request(config):
    mock = MockMusic()
    production = sample_production()
    production.scenes[0].mood = "urgent"
    with FilmRenderer(
        config,
        registry=BackendRegistry([ProceduralBackend(config)]),
        music_registry=MusicRegistry([mock]),
    ) as film:
        film.render(production)
    assert "urgent" in mock.requests[0].moods


# --------------------------------------------------------------------------- #
# Hosted speech
# --------------------------------------------------------------------------- #


def test_a_hosted_voice_round_trips_through_the_screenplay():
    profile = VoiceProfile(voice="Kathy", hosted_voice="abc123")
    assert VoiceProfile.from_dict(profile.to_dict()).hosted_voice == "abc123"


def test_an_unset_hosted_voice_is_not_written_to_the_file():
    """Every screenplay written before this field existed must round-trip
    unchanged, byte for byte."""
    assert "hosted_voice" not in VoiceProfile(voice="Kathy").to_dict()
    production = sample_production()
    cast_production(production.cast, POOL)
    assert "hosted_voice" not in production.to_markdown()


def test_hosted_voices_are_dealt_round_robin_and_deterministically():
    cast = [Character("Maya"), Character("Jonas"), Character("Ada")]
    cast_production(cast, POOL, hosted_voices=["v1", "v2"])
    assigned = {c.name: c.voice.hosted_voice for c in cast}
    assert set(assigned.values()) == {"v1", "v2"}

    again = [Character("Maya"), Character("Jonas"), Character("Ada")]
    cast_production(again, POOL, hosted_voices=["v1", "v2"])
    assert {c.name: c.voice.hosted_voice for c in again} == assigned


def test_a_system_voice_is_cast_alongside_the_hosted_one():
    """Switching the paid tier back off must not mute the cast."""
    cast = [Character("Maya"), Character("Jonas")]
    cast_production(cast, POOL, hosted_voices=["v1", "v2"])
    assert all(c.voice.voice for c in cast)
    assert all(c.voice.hosted_voice for c in cast)


def test_a_hand_set_hosted_voice_is_never_overwritten():
    cast = [Character("Maya", voice=VoiceProfile(voice="Kathy", hosted_voice="mine"))]
    cast_production(cast, POOL, hosted_voices=["v1"])
    assert cast[0].voice.hosted_voice == "mine"


def test_hosted_speech_is_off_without_a_config():
    assert SpeechSynthesizer().hosted is None
    assert SpeechSynthesizer(config=GenerationConfig()).hosted is None


def test_hosted_speech_is_off_without_a_key(tmp_path):
    spec = providers.ProviderSpec(
        name="tts", tier=providers.SPEECH, endpoint="https://e",
        api_key_env="NOT_SET_ANYWHERE_98765",
    )
    config = GenerationConfig(cache_dir=tmp_path)
    config._providers = providers.ProvidersConfig(
        enabled={providers.MUSIC: False, providers.SPEECH: True},
        selected={providers.MUSIC: "", providers.SPEECH: "tts"},
        specs={"tts": spec},
    )
    assert SpeechSynthesizer(config=config).hosted is None


def hosted_speech_config(tmp_path, voices=("v1", "v2")):
    spec = providers.ProviderSpec(
        name="tts",
        tier=providers.SPEECH,
        endpoint="https://example.test/tts/{voice}",
        api_key="k",
        auth_header="x-api-key",
        auth_prefix="",
        body={"text": "{text}"},
        response="audio",
        voices=list(voices),
    )
    config = GenerationConfig(
        preset="draft", resolution=(320, 180), fps=12,
        output_dir=tmp_path / "o", scripts_dir=tmp_path / "s",
        characters_dir=tmp_path / "c", cache_dir=tmp_path / "k",
    )
    config._providers = providers.ProvidersConfig(
        enabled={providers.MUSIC: False, providers.SPEECH: True},
        selected={providers.MUSIC: "", providers.SPEECH: "tts"},
        specs={"tts": spec},
    )
    return config


def test_hosted_speech_speaks_a_line_and_uses_the_characters_voice_id(
    tmp_path, wav_bytes, monkeypatch
):
    config = hosted_speech_config(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse(wav_bytes, "audio/wav")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    speech = SpeechSynthesizer(allow_neural=False, config=config)
    assert speech.hosted is not None

    path, duration = speech.speak(
        "Hello there.", VoiceProfile(hosted_voice="voice-42"), tmp_path / "a.wav"
    )
    assert captured["url"] == "https://example.test/tts/voice-42"
    assert captured["body"]["text"] == "Hello there."
    assert duration > 0 and Path(path).is_file()
    assert speech.voiced_lines == 1 and speech.silent_lines == 0
    assert speech.engine_name == "hosted:tts"


def test_hosted_speech_falls_back_to_the_system_voice_on_failure(tmp_path, monkeypatch):
    """A provider outage must not mute the film."""
    import urllib.error

    config = hosted_speech_config(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    speech = SpeechSynthesizer(allow_neural=False, config=config)
    if speech.command is None:
        pytest.skip("no system synthesiser to fall back to")
    speech.speak("Hello there.", VoiceProfile(voice=POOL[0].name), tmp_path / "a.wav")
    assert speech.voiced_lines == 1
    assert speech.engine_name == speech.command
    assert any("down" in f for f in speech.failures)


@pytest.mark.slow
def test_one_hosted_failure_disables_the_tier_for_the_rest_of_the_render(tmp_path, monkeypatch):
    """Paying for a dead provider once per line is worse than falling back."""
    import urllib.error

    config = hosted_speech_config(tmp_path)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    speech = SpeechSynthesizer(allow_neural=False, config=config)
    for index in range(4):
        speech.speak("A line.", VoiceProfile(voice=POOL[0].name), tmp_path / f"{index}.wav")
    assert len(calls) == 1
    assert speech.hosted is None


# --------------------------------------------------------------------------- #
# Two defects found while writing the worked examples
# --------------------------------------------------------------------------- #


def test_a_screenplay_is_not_titled_INT():
    """`split_sentences` ends the first sentence at the abbreviation's full
    stop, so every script used to be called "INT"."""
    script = "INT. CONTROL ROOM - NIGHT\n\nMAYA: Hello.\n"
    assert Screenwriter.derive_title(script) == "Control Room"
    assert Screenwriter.derive_title("EXT. CAR PARK - DAY\n\nJONAS: Hi.\n") == "Car Park"


def test_an_explicit_title_still_wins_over_the_slugline():
    script = "Title: The Signal\n\nINT. CONTROL ROOM - NIGHT\n\nMAYA: Hello.\n"
    assert Screenwriter.derive_title(script) == "The Signal"


def test_prose_and_briefs_are_titled_as_before():
    """The slugline path must not capture anything that is not a screenplay."""
    assert Screenwriter.derive_title("Maya ran the night shift.") == "Maya ran the night shift"
    brief = "A 20 second vertical advert for Kestrel Coffee, a small roastery."
    assert Screenwriter.derive_title(brief).startswith("Kestrel Coffee")


def test_a_slugline_with_no_location_falls_through():
    assert Screenwriter.derive_title("INT.\n\nMAYA: Hello.\n") != ""


def test_a_wrapped_line_does_not_carry_its_line_break_into_a_caption():
    """Prose wrapped at 80 columns rendered a stray break mid-sentence."""
    line = Line("Maya", "We are out\nof time, and the\ttrace is climbing.")
    assert line.text == "We are out of time, and the trace is climbing."
    assert "\n" not in line.text


def test_whitespace_is_normalised_in_delivery_too():
    assert Line("Maya", "Hi.", delivery="very\n quietly").delivery == "very quietly"


def test_narration_parsed_from_wrapped_prose_is_clean():
    prose = (
        "Maya had run the night shift for three years, and she had never seen\n"
        "the trace climb like this. Jonas leaned over the console beside her.\n"
    )
    production = Screenwriter().write(prose, StoryMeta(title="Wrapped"))
    texts = [line.text for shot in production.shots for line in shot.lines]
    assert texts
    assert all("\n" not in text for text in texts)
