"""THE RUN ON MODAL: the sealed runner (run_trajmaster.py, unchanged) on one A100-80GB; the model's weights (the pinned
revision, about 23 GB: no optimizer states, no second copy of the base transformer) cached in a Modal volume.  From
this folder, after `pip install modal` and `modal setup`:
    modal run modal_app.py                        control, slide and jump
    modal run modal_app.py --shots slide          one shot
The videos and run.json come back into ../out/.  Nothing else is sent: inputs.json is the sealed file."""
import os, pathlib
import modal

HERE = pathlib.Path(__file__).parent
CODE = "0af2932ffe514f0842bce3a2ef8809773661f8ee"              # github.com/KwaiVGI/3DTrajMaster
WEIGHTS = "d38a0c8fe2bdcf0ae0dfc98d1fff3bd0be5bd102"           # huggingface.co/KlingTeam/3DTrajMaster

image = (modal.Image.debian_slim(python_version="3.11")          # Modal's 2026.10 builder starts at 3.11
         .apt_install("git", "ffmpeg")
         .pip_install("torch==2.5.1", "diffusers==0.31.0", "transformers==4.46.2", "accelerate==1.1.1", "peft==0.12.0",
                      "sentencepiece==0.2.0", "einops==0.8.0", "imageio==2.35.1", "imageio-ffmpeg==0.5.1", "numpy==1.26.4",
                      "huggingface_hub==0.26.2", "hf_transfer==0.1.8", "safetensors==0.4.5")
         .run_commands(f"git clone https://github.com/KwaiVGI/3DTrajMaster /opt/3DTrajMaster && cd /opt/3DTrajMaster && git checkout {CODE}")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
         .add_local_file(HERE / "run_trajmaster.py", "/root/run_trajmaster.py"))
vol = modal.Volume.from_name("am-trajmaster-weights", create_if_missing=True)
app = modal.App("am-numbers-test", image=image)


@app.function(volumes={"/weights": vol}, timeout=3600, cpu=4, memory=16384)
def fetch():
    from huggingface_hub import snapshot_download
    done = "/weights/%s.done" % WEIGHTS
    if os.path.exists(done):
        return "weights cached"
    snapshot_download("KlingTeam/3DTrajMaster", revision=WEIGHTS, local_dir="/weights/3dtm",
                      allow_patterns=["cogvideox-5b/model_index.json", "cogvideox-5b/scheduler/*", "cogvideox-5b/text_encoder/*",
                                      "cogvideox-5b/tokenizer/*", "cogvideox-5b/vae/*", "cogvideox-5b/transformer/config.json",
                                      "injector/config.json", "injector/diffusion_pytorch_model*",
                                      "lora/pytorch_lora_weights.safetensors"])
    open(done, "w").write("ok")
    vol.commit()
    return "weights downloaded"


@app.function(gpu="A100-80GB", volumes={"/weights": vol}, timeout=3600, cpu=8, memory=65536)   # the runner copies 12.7 GB on the CPU
def generate(inputs: str, shots: str) -> dict:
    import subprocess
    open("/root/inputs.json", "w").write(inputs)
    subprocess.run(["python", "/root/run_trajmaster.py", "--repo", "/opt/3DTrajMaster", "--weights", "/weights/3dtm",
                    "--inputs", "/root/inputs.json", "--out", "/root/out", "--shots", shots], check=True)
    return {f: open(os.path.join("/root/out", f), "rb").read() for f in os.listdir("/root/out")}


@app.local_entrypoint()
def main(shots: str = "control,slide,jump"):
    print(fetch.remote())
    files = generate.remote((HERE.parent / "inputs.json").read_text(), shots)
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
        print("got", name, len(data), "bytes")
