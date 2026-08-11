"""Voice discovery and character-to-voice casting.

The pool is whatever the host actually offers -- on macOS that is the ~40
English `say` voices, which cover several genders, ages and accents well enough
to cast a small film. Casting is deterministic and written back to the character
file, so a character never changes voice between renders.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .model import Character, VoiceProfile, stable_seed

LOGGER = logging.getLogger("video_gen.casting")


@dataclass(frozen=True)
class SystemVoice:
    """One voice offered by the host synthesiser."""

    name: str
    locale: str = "en_US"
    gender: str = "unknown"
    age: str = "adult"
    novelty: bool = False  # singing/robot voices: unusable as a normal performer

    @property
    def accent(self) -> str:
        return self.locale.split("_")[-1].lower()


# Gender/age for the stock macOS voices. The OS does not expose this metadata,
# and casting a story's grandmother as a child voice is worse than any of the
# fallbacks, so the well-known ones are tabulated. Anything absent is still
# usable -- it simply casts as an unknown-gender adult.
_VOICE_TRAITS: Dict[str, Dict[str, object]] = {
    "samantha": {"gender": "female", "age": "adult"},
    "kathy": {"gender": "female", "age": "adult"},
    "karen": {"gender": "female", "age": "adult"},
    "moira": {"gender": "female", "age": "adult"},
    "tessa": {"gender": "female", "age": "adult"},
    "fiona": {"gender": "female", "age": "adult"},
    "veena": {"gender": "female", "age": "adult"},
    "allison": {"gender": "female", "age": "adult"},
    "ava": {"gender": "female", "age": "adult"},
    "susan": {"gender": "female", "age": "adult"},
    "vicki": {"gender": "female", "age": "adult"},
    "victoria": {"gender": "female", "age": "adult"},
    "flo": {"gender": "female", "age": "adult"},
    "shelley": {"gender": "female", "age": "adult"},
    "nora": {"gender": "female", "age": "adult"},
    "alex": {"gender": "male", "age": "adult"},
    "daniel": {"gender": "male", "age": "adult"},
    "fred": {"gender": "male", "age": "adult"},
    "rishi": {"gender": "male", "age": "adult"},
    "tom": {"gender": "male", "age": "adult"},
    "aaron": {"gender": "male", "age": "adult"},
    "ralph": {"gender": "male", "age": "adult"},
    "reed": {"gender": "male", "age": "adult"},
    "rocko": {"gender": "male", "age": "adult"},
    "eddy": {"gender": "neutral", "age": "adult"},
    "sandy": {"gender": "neutral", "age": "adult"},
    "junior": {"gender": "male", "age": "child"},
    "grandma": {"gender": "female", "age": "elder"},
    "grandpa": {"gender": "male", "age": "elder"},
    "albert": {"gender": "male", "age": "adult", "novelty": True},
    "whisper": {"gender": "male", "age": "adult", "novelty": True},
}

#: Voices that sing, buzz or warble. Kept out of ordinary casting, but still
#: available by name for robots, monsters and mascots.
_NOVELTY = {
    "bad news", "good news", "bahh", "bells", "boing", "bubbles", "cellos",
    "jester", "organ", "superstar", "trinoids", "wobble", "zarvox", "albert",
    "whisper", "deranged", "hysterical", "pipe organ", "princess", "bruce",
}

# Common given names, so a script that supplies nothing but "MAYA:" still casts
# sensibly. Deliberately not exhaustive -- it is a prior, overridden by anything
# the description says, and by an explicit voice in the character file.
_FEMALE_NAMES = (
    "maya mia emma olivia ava sophia isabella charlotte amelia harper evelyn abigail emily "
    "elizabeth sofia ella grace chloe victoria riley aria lily nora hazel zoe eleanor clara "
    "sarah anna maria laura julia rachel hannah rebecca alice rose ruby ivy jade nina lena "
    "priya aisha fatima mei yuki elena irina katya leila amara zara naomi tara ada agnes "
    "diana helen susan linda nancy betty dorothy carol sandra donna edith freya sana"
).split()

_MALE_NAMES = (
    "jonas james john robert michael william david richard joseph thomas charles chris daniel "
    "matthew anthony mark donald steven paul andrew joshua kenneth kevin brian george edward "
    "ronald timothy jason jeffrey ryan jacob gary nicholas eric jonathan stephen larry justin "
    "samuel liam noah oliver elijah lucas mason logan ethan aiden henry sebastian jack owen "
    "arjun rahul wei hiroshi omar hassan ivan dmitri carlos diego marco luca felix victor "
    "walter arthur albert frank harold roy eugene louis philip malik tomas"
).split()

_GENDER_BY_NAME: Dict[str, str] = {name: "female" for name in _FEMALE_NAMES}
_GENDER_BY_NAME.update({name: "male" for name in _MALE_NAMES if name not in _GENDER_BY_NAME})

_FEMALE_HINTS = re.compile(
    r"\b(she|her|hers|woman|women|girl|lady|mother|mom|mum|daughter|sister|aunt|"
    r"grandmother|grandma|wife|queen|princess|actress|female|madam|ms|mrs|miss)\b",
    re.I,
)
_MALE_HINTS = re.compile(
    r"\b(he|him|his|man|men|boy|guy|father|dad|son|brother|uncle|grandfather|"
    r"grandpa|husband|king|prince|actor|male|sir|mr)\b",
    re.I,
)
# "young" is deliberately absent: "a young woman" is an adult, and matching it
# here casts leads as children.
_CHILD_HINTS = re.compile(
    r"\b(child|kid|kids|boy|girl|teen|teenager|toddler|schoolgirl|schoolboy|infant|"
    r"son|daughter|pupil)\b",
    re.I,
)
_ELDER_HINTS = re.compile(r"\b(old|elderly|aged|ancient|grandmother|grandma|grandfather|grandpa|veteran|retired)\b", re.I)

_ACCENT_HINTS = {
    "gb": re.compile(r"\b(british|english|london|uk|cockney|posh)\b", re.I),
    "au": re.compile(r"\b(australian|aussie|sydney|melbourne)\b", re.I),
    "ie": re.compile(r"\b(irish|ireland|dublin)\b", re.I),
    "in": re.compile(r"\b(indian|india|mumbai|delhi)\b", re.I),
    "za": re.compile(r"\b(south african|johannesburg|cape town)\b", re.I),
    "us": re.compile(r"\b(american|america|new york|texas|california)\b", re.I),
}


def system_tts_command() -> Optional[str]:
    """The host's offline speech synthesiser, if any."""
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for candidate in ("espeak-ng", "espeak"):
        if shutil.which(candidate):
            return candidate
    return None


def _traits_for(name: str) -> Dict[str, object]:
    base = name.split("(")[0].strip().lower()
    traits = dict(_VOICE_TRAITS.get(base, {}))
    if base in _NOVELTY:
        traits["novelty"] = True
    return traits


def discover_voices(language: str = "en") -> List[SystemVoice]:
    """Enumerate the host's voices, annotated with what we know about them."""
    command = system_tts_command()
    if command is None:
        return []
    if command != "say":
        # espeak exposes a very different listing; one neutral voice is enough
        # to keep multi-character output working on Linux.
        return [SystemVoice(name="default", locale="en_US", gender="neutral")]
    try:
        output = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=20, check=True
        ).stdout
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not enumerate voices: %s", exc)
        return []

    voices: List[SystemVoice] = []
    for row in output.splitlines():
        row = row.split("#")[0].rstrip()
        if not row:
            continue
        parts = row.split()
        if len(parts) < 2:
            continue
        locale = parts[-1]
        name = " ".join(parts[:-1]).strip()
        if not name or not locale.lower().startswith(language.lower()):
            continue
        traits = _traits_for(name)
        voices.append(
            SystemVoice(
                name=name,
                locale=locale,
                gender=str(traits.get("gender", "unknown")),
                age=str(traits.get("age", "adult")),
                novelty=bool(traits.get("novelty", False)),
            )
        )
    return voices


def infer_traits(character: Character) -> Dict[str, str]:
    """Guess gender/age/accent from how a character is written.

    Description text is the strongest signal, but a screenplay often gives a
    character nothing but a name -- and casting "MAYA" with a male voice is a
    glaring error, so the given-name table backs the text heuristics up.
    """
    text = " ".join(
        filter(None, [character.role, character.description, character.appearance])
    )
    female = len(_FEMALE_HINTS.findall(text))
    male = len(_MALE_HINTS.findall(text))
    gender = "unknown"
    if female > male:
        gender = "female"
    elif male > female:
        gender = "male"
    if gender == "unknown":
        gender = _GENDER_BY_NAME.get(character.name.strip().split()[0].lower(), "unknown")
    text = f"{character.name} {text}"

    age = "adult"
    if _ELDER_HINTS.search(text):
        age = "elder"
    elif _CHILD_HINTS.search(text):
        age = "child"

    accent = ""
    for code, pattern in _ACCENT_HINTS.items():
        if pattern.search(text):
            accent = code
            break
    return {"gender": gender, "age": age, "accent": accent}


def _score(voice: SystemVoice, traits: Dict[str, str]) -> float:
    """How well a voice fits a character. Higher is better."""
    score = 0.0
    if traits["gender"] != "unknown":
        if voice.gender == traits["gender"]:
            score += 6.0
        elif voice.gender == "neutral":
            score += 1.5
        elif voice.gender != "unknown":
            score -= 5.0
    if voice.age == traits["age"]:
        score += 4.0
    elif traits["age"] != "adult" or voice.age != "adult":
        score -= 2.5
    if traits["accent"]:
        score += 3.0 if voice.accent == traits["accent"] else 0.0
    elif voice.accent == "us":
        score += 0.75  # mild default so casting is stable, not arbitrary
    if voice.novelty:
        score -= 12.0
    if voice.gender == "unknown":
        score -= 1.0
    return score


def cast_voice(
    character: Character,
    pool: Sequence[SystemVoice],
    taken: Optional[Sequence[str]] = None,
) -> VoiceProfile:
    """Pick the best unused voice for a character, then shape it.

    Distinctness is enforced twice: the best-fitting *unused* voice is preferred,
    and pitch/rate are nudged per character so that even a reused voice does not
    sound like the same performer.
    """
    traits = infer_traits(character)
    taken_set = {t.lower() for t in (taken or [])}

    if not pool:
        return VoiceProfile(
            engine="silent", voice="", gender=traits["gender"], age=traits["age"]
        )

    seed = stable_seed(character.key)
    ranked = sorted(
        pool,
        key=lambda v: (
            -(_score(v, traits) + (0.0 if v.name.lower() in taken_set else 3.0)),
            # Deterministic, character-dependent tiebreak: two similar
            # characters do not both land on the same "first alphabetically".
            stable_seed(v.name, seed) % 1000,
        ),
    )
    chosen = ranked[0]

    # Shaping: deterministic per character, small enough to stay natural.
    rate = 1.0 + ((seed % 21) - 10) / 100.0  # +/- 10%
    pitch = ((stable_seed(character.key, 7) % 9) - 4) * 0.9  # +/- ~3.6 semitones
    return VoiceProfile(
        engine="system",
        voice=chosen.name,
        rate=round(rate, 3),
        pitch=round(pitch, 2),
        gender=chosen.gender if chosen.gender != "unknown" else traits["gender"],
        age=chosen.age,
    )


def cast_production(characters: Sequence[Character], pool: Optional[Sequence[SystemVoice]] = None) -> None:
    """Assign a voice to every character that does not already have one.

    Characters are cast in a stable order so the result does not depend on the
    order they happened to be discovered in the script.
    """
    voices = list(pool if pool is not None else discover_voices())
    ordered = sorted(characters, key=lambda c: c.key)
    taken = [c.voice.voice for c in characters if c.voice.voice]
    for character in ordered:
        if character.voice.voice:
            continue  # already cast, possibly by hand -- never override
        profile = cast_voice(character, voices, taken=taken)
        character.voice = profile
        if profile.voice:
            taken.append(profile.voice)


def describe_pool(pool: Optional[Sequence[SystemVoice]] = None) -> str:
    """One-line human summary of what this machine can cast from."""
    voices = list(pool if pool is not None else discover_voices())
    if not voices:
        return "no system voices found (narration will be silent with captions)"
    usable = [v for v in voices if not v.novelty]
    accents = sorted({v.accent.upper() for v in usable})
    return (
        f"{len(usable)} castable voices "
        f"({sum(v.gender == 'female' for v in usable)}F / "
        f"{sum(v.gender == 'male' for v in usable)}M), accents: {', '.join(accents)}"
    )
