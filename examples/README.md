# Worked examples

Four input shapes, one agent loop. Everything here runs offline and needs no key.

Run them from the repository root.

```bash
pip install -e .
python -m video_gen --doctor        # confirm this machine can speak
```

---

## 1. A screenplay — the reliable path

Sluglines and `NAME:` lines are parsed directly, including parentheticals. Use
this whenever you know what should be said.

```bash
python -m video_gen --prompt-file examples/01_screenplay.txt \
    --format short --output output/example-screenplay.mp4 --yes
```

Expect: title `Control Room` (from the first slugline), 2 scenes, 5 shots,
2 characters with different voices, ~12 s at 1280×720.

Each line becomes its own shot, so the camera can cut between speakers. That is
why 5 lines produce 5 shots rather than 2.

---

## 2. Prose — speech lifted out of narration

Sentences become beats, quoted speech becomes attributed dialogue, and pronouns
resolve to the right character.

```bash
python -m video_gen --prompt-file examples/02_prose.txt \
    --format short --output output/example-prose.mp4 --yes
```

Expect: `MAYA: We are out of time.` attributed to Maya — not to the narrator and
not to Jonas — and `JONAS: Then it was never noise.` from the second paragraph.
Everything unquoted stays narration.

---

## 3. A brief — structured and staged, reported speech promoted

```bash
python -m video_gen --prompt-file examples/03_brief.txt \
    --format ad --aspect 9:16 --seconds 20 \
    --output output/example-brief.mp4 --yes
```

Expect: cast `Ama` and `Dev` — **not** "Kestrel" and "Coffee" — a true 720×1280
frame, and `AMA: The first cup is on the house.` lifted out of the instruction.

One shot, because one line of speech was recoverable from the brief. Add `--llm`
to get the rest of the ad written, or pass a fuller script.

The brand is deliberately kept out of the cast. This is the case most text-to-
video tools get wrong.

---

## 4. A bare topic — needs a local model

```bash
# Without a model: one shot of the narrator reading your prompt back.
python -m video_gen --prompt-file examples/04_topic.txt \
    --format educational --seconds 30 --output output/example-topic.mp4 --yes

# With one: invented characters having a conversation.
ollama pull llama3.2:1b
python -m video_gen --prompt-file examples/04_topic.txt \
    --format educational --seconds 30 --llm \
    --output output/example-topic-llm.mp4 --yes
```

The first command behaving that way is by design, not a bug: regular
expressions cannot invent speech. Check `capabilities.write_dialogue` in
`--doctor --json` before relying on the second.

`llama3.2:1b` is the floor, not the ceiling. It writes serviceable copy and
sometimes drifts off-brief, and it will not reliably produce a two-hander — a
one-character monologue is a common and acceptable result. Any larger Ollama
model is a drop-in upgrade via `--llm-model`.

---

## 5. The agent loop

```bash
python examples/agent_workflow.py
python examples/agent_workflow.py --prompt-file examples/03_brief.txt
```

[`agent_workflow.py`](agent_workflow.py) is the shape to copy: probe
capabilities, decide whether a model is needed, write the screenplay with
`--no-render`, read it back with `--show`, then render and check the exit code.

It parses only JSON, never console output, and it treats exit code **2**
("rendered but no line was audible") as a reportable failure rather than a
success — because that file looks finished on disk and plays subtitles over
silence.

---

## Editing and re-rendering

Every run writes a screenplay to `video_scripts/<id>.md`. Edit a line and
re-render; images are cached by prompt, preset and seed, so only what changed
regenerates.

```bash
python -m video_gen --list                 # find the id
python -m video_gen --show <id>            # read it
$EDITOR video_scripts/<id>.md              # change a line
python -m video_gen --render <id> --yes    # re-render
python -m video_gen --cast <id>            # re-roll the voices instead
```

## What the output looks like

Without the optional model stack the visuals are composed environments with
staged geometric figures — stylised, deliberately, with no lip sync and no limb
animation. The screenplay, the casting, the per-character voices, the captions
and the ducked score are all fully working. Set expectations on that before
handing a file to anyone.
