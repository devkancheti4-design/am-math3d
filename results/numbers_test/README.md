# The numbers test: AM's numbers straight into 3DTrajMaster

**Status (3 Nov 2026): sealed, not run.** Making the videos needs a CUDA GPU with 40 GB or more; the run is waiting
for one.

## The question

Can a model that takes numbers instead of pictures follow AM's numbers? Here AM writes no guide video. It writes the
13 poses that [3DTrajMaster](https://github.com/KwaiVGI/3DTrajMaster) (ICLR 2025) takes as its input. The open release
of 3DTrajMaster is CogVideoX-5B with a LoRA and a pose injector.

## Measured already

| | median error | where it comes from |
|---|---|---|
| The model's training renders against their own numbers | 10.7 px | `data/contract.json` |
| The model's 10 published one-entity videos against their own numbers | 29.0 px (about 21 cm) | `data/baseline.json` |
| The pass bar for AM's shots | 44.7 px | the 90th percentile of those 10 |

The camera that 3DTrajMaster's pose numbers are written for is not the camera that rendered its training clips. That
camera stands about 3.45 m higher and looks 3.3° further down. Projected through the published camera, the numbers
miss the animals by about 400 px. `tools/make_contract.py` fits the real camera from the model's own renders: the
contract.

## Sealed

The lock is `61c9ceb05e48f76d064cb354ae9adc1b1b35a127ae21a45a1d831eac0515d95d`, written 3 Nov 2026 at 08:05 IST.
`lock.json` lists what it covers:

- the inputs and the rules;
- the contract, the baseline and the ground;
- AM's two plans and the control's;
- the scorer, AM's intent solver and the runner.

The three shots:

- **control:** the model's own published robot shot, with its own poses and words. It checks the runner.
- **slide:** the fox stands for 0.3 s, then slides 3.6 m with e = 3 s² − 2 s³. It starts and stops at rest and lands
  on AM's spot at frame 39.
- **jump:** the fox stands for 1.0 s, then leaps 2.3 m with a 70 cm gravity arc. It lands at frame 30. No training
  entity ever leaves the ground, so this one is a stretch.

The words are the model's own template, "<entity> is moving in the <location>". The rules are in `RULES.md`.

## To finish it

Use any machine with a CUDA GPU of 40 GB or more:

```bash
git clone https://github.com/KwaiVGI/3DTrajMaster && git -C 3DTrajMaster checkout 0af2932ffe514f0842bce3a2ef8809773661f8ee
pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.2 accelerate==1.1.1 peft==0.12.0 sentencepiece==0.2.0 einops imageio==2.35.1 imageio-ffmpeg==0.5.1
huggingface-cli download KlingTeam/3DTrajMaster --revision d38a0c8fe2bdcf0ae0dfc98d1fff3bd0be5bd102 --local-dir weights --exclude "*optimizer.bin" "cogvideox-5b/transformer/*.safetensors"
python3 run/run_trajmaster.py --repo 3DTrajMaster --weights weights --inputs inputs.json --out out
```

On Modal, `modal run run/modal_app.py` does all four steps.

Then score the videos on this Mac with `python3 tools/score_shots.py`. It checks the lock first, then writes
`out/<shot>/score.json`, `out/<shot>/labelled.mp4` (AM's pixel and the animal's on every frame) and `out/verdicts.json`.
