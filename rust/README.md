# video-gen-core

Optional native accelerator for [`video_gen_ai`](https://github.com/minjaezekiel/video_gen_ai).

```bash
pip install ./rust        # from a checkout
```

`video_gen` works without it. `video_gen.fastpath` imports this if present and
falls back to NumPy and Pillow if not, and the test suite checks both
implementations produce the same pixels.

It adds no capability. It makes three measured hot loops cheaper:

| function | why |
|---|---|
| `resize_crop` | the Ken Burns crop + resize, once per frame — 20% of a render |
| `blend_box` | the caption alpha composite, once per captioned frame |
| `mouth_openness` | the lip-sync envelope, once per shot |

Built with PyO3 and rayon as an abi3 wheel, so one build covers CPython 3.9+.
