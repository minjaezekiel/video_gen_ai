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
