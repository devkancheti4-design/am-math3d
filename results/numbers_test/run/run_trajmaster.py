"""THE RUN: 3DTrajMaster's own inference (CogVideo/inference/3dtrajmaster_inference.py at the pinned commit), with one
change: the poses come from inputs.json (written and sealed before this runs) instead of its dataset.  Everything else
is the README's command: CogVideoX-5B with the injector, the LoRA fused at 0.6, the injector kept for the first 20 of
50 DPM steps (then the copy without the LoRA, poses off), guidance 6 with the dynamic schedule, 49 frames, 720 x 480,
bfloat16, seed per shot on a CPU generator, the video written at 8 fps.  Needs one CUDA GPU (40 GB or more).
    python3 run_trajmaster.py --repo 3DTrajMaster --weights weights --inputs inputs.json --out out [--shots control,slide,jump]
<weights>: KlingTeam/3DTrajMaster at the pinned revision: cogvideox-5b/ (text encoder, VAE, tokenizer, scheduler, configs),
injector/ (the transformer with the injector), lora/pytorch_lora_weights.safetensors."""
import argparse, copy, json, os, platform, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shots", default="control,slide,jump")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "CogVideo", "finetune"))

import torch                                                    # noqa: E402
from diffusers import CogVideoXDPMScheduler                     # noqa: E402
from diffusers.utils import export_to_video                     # noqa: E402
from models.pipeline_cogvideox import CogVideoXPipeline         # noqa: E402  (the model's own pipeline)
from models.cogvideox_transformer_3d import CogVideoXTransformer3DModel   # noqa: E402

inp = json.load(open(a.inputs))
S = inp["model"]["settings"]
os.makedirs(a.out, exist_ok=True)
rec = dict(gpu=torch.cuda.get_device_name(0), gpu_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
           torch=torch.__version__, python=platform.python_version(), settings=S, shots={})
t0 = time.time()
dtype = torch.bfloat16
transformer = CogVideoXTransformer3DModel.from_pretrained(os.path.join(a.weights, "injector"), torch_dtype=dtype)
pipe = CogVideoXPipeline.from_pretrained(os.path.join(a.weights, "cogvideox-5b"), transformer=transformer, torch_dtype=dtype)
pipe.transformer_ori = copy.deepcopy(pipe.transformer).to("cuda")
pipe.load_lora_weights(os.path.join(a.weights, "lora"), weight_name="pytorch_lora_weights.safetensors", adapter_name="default")
pipe.fuse_lora(components=["transformer"], lora_scale=S["lora_scale"])
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
if rec["gpu_gb"] >= 60:
    pipe.to("cuda")                                             # everything fits: same numbers, no offload
else:
    pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
rec["load_seconds"] = round(time.time() - t0, 1)
import diffusers, transformers                                  # noqa: E402
rec["diffusers"], rec["transformers"] = diffusers.__version__, transformers.__version__

for shot in inp["shots"]:
    if shot["id"] not in a.shots.split(","):
        continue
    t1 = time.time()
    pose = torch.tensor(shot["pose_embeds"], dtype=torch.float32).to(dtype).to("cuda")    # (entities, 13, 12)
    frames = pipe(prompt=shot["prompt"], prompts_list=shot["entity_prompts"], pose_embeds=pose[None],
                  num_videos_per_prompt=1, annealed_sample_step=S["annealed_sample_step"],
                  num_inference_steps=S["num_inference_steps"], num_frames=S["num_frames"],
                  use_dynamic_cfg=S["use_dynamic_cfg"], guidance_scale=S["guidance_scale"],
                  generator=torch.Generator().manual_seed(shot["seed"])).frames[0]
    path = os.path.join(a.out, shot["id"] + ".mp4")
    export_to_video(frames, path, fps=S["export_fps"])
    rec["shots"][shot["id"]] = dict(seconds=round(time.time() - t1, 1), frames=len(frames), file=os.path.basename(path),
                                    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    print(shot["id"], rec["shots"][shot["id"]], flush=True)
    json.dump(rec, open(os.path.join(a.out, "run.json"), "w"), indent=1)
rec["total_seconds"] = round(time.time() - t0, 1)
json.dump(rec, open(os.path.join(a.out, "run.json"), "w"), indent=1)
print("done", rec["total_seconds"], "s")
