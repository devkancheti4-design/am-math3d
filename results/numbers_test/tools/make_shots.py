"""As run (3 Nov 2026): AM's two shots for 3DTrajMaster, the control shot, and the lock, all before anything is generated.
    python3 make_shots.py <360Motion root>
  SLIDE: the fox stands 0.3 s, slides 3.6 m on the ground (e = 3 s^2 - 2 s^3: starts and stops at rest) and stops on
    AM's spot at frame 39, then stands to the end.
  JUMP: the fox stands 1.0 s, leaps 2.3 m with a 70 cm gravity arc (13 airborne frames, every second difference the same)
    and lands on AM's spot at frame 30, then stands.  None of the model's training entities ever leaves the ground.
  CONTROL: the model's own published shot D_loc2_17 (the rescue robot on the sunset beach), its own poses read by its
    own code, seed 24: the runner has to reproduce the model before AM's shots mean anything.
The words are the model's own template, '<entity> is moving in the <location>': nothing about stopping, jumping, timing
or place.  All of that is only in the numbers."""
import hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TEST, "..", "..", "math3d"))
import tjm_geom as G                                            # noqa: E402
import am_numbers as N                                          # noqa: E402
import am_intent                                                # noqa: E402

root = sys.argv[1]
data = os.path.join(TEST, "data")
con = N.contract(os.path.join(data, "contract.json"))
ground = json.load(open(os.path.join(data, "ground.json")))
rule_px = json.load(open(os.path.join(data, "baseline.json")))["summary"]["rule_px"]
FOX = "a fox with sleek russet fur, a bushy tail tipped with black, and bright green and cunning eyes"   # the model's entity zoo
SIZE = (80.0, 80.0, 70.0)                                       # across, forward_seen, up (cm): only for AM's checks
SETTINGS = dict(lora_scale=0.6, annealed_sample_step=20, num_inference_steps=50, guidance_scale=6.0, use_dynamic_cfg=True,
                num_frames=49, height=480, width=720, dtype="bfloat16", scheduler="CogVideoXDPMScheduler, timestep_spacing=trailing",
                export_fps=8, note="the README's inference command, unchanged")

shots, plans = [], {}
for sid, start, ask in (("slide", (-150.0, 120.0), dict(target=dict(across=120.0, forward=-120.0), mode="slide", duration=2.0, wait=0.3)),
                        ("jump", (-120.0, 50.0), dict(target=dict(across=90.0, forward=-50.0), mode="jump", height=70.0, wait=1.0))):
    wdir = os.path.join(TEST, sid)
    N.world(wdir, con, ground, start, SIZE)
    t0 = time.time()
    plan = am_intent.solve(wdir, 1, ask["target"], height=ask.get("height"), wait=ask["wait"], seconds=N.FRAMES / N.FPS,
                           mode=ask["mode"], duration=ask.get("duration"), write=False)
    assert plan["verdict"] == "SOLVED" and len(plan["rows"]) == N.FRAMES, (sid, plan["verdict"], plan["why"])
    m = N.to_model(plan, con)
    back = N.lens(N.to_camera(m["numbers"], con))                # the numbers, read back through the contract
    assert np.abs(back - N.expected(plan)).max() < 0.05, np.abs(back - N.expected(plan)).max()
    plan["model_input"] = dict(pose_embeds=np.round(m["pose_embeds"], 6).tolist(), latent_am_frames=m["latent_am_frames"],
                               latent_clip_frames=m["latent_clip_frames"], facing=np.round(m["facing"], 6).tolist(),
                               numbers=np.round(m["numbers"], 5).tolist(), solve_seconds=round(time.time() - t0, 4))
    json.dump(plan, open(os.path.join(wdir, "plan.json"), "w"), indent=1)
    plans[sid] = plan
    shots.append(dict(id=sid, entity_prompts=[FOX], loc_prompt="desert", prompt=FOX + " is moving in the desert",
                      pose_embeds=plan["model_input"]["pose_embeds"], seed=24))
    a = plan["answer"]
    print("%-5s %s: take-off frame %d, %d frames moving, lands frame %d; %s" % (sid, plan["verdict"], a["take_off"], a["air_frames"],
          a["take_off"] + a["air_frames"], a["equation"]))
    print("      AM's pixels: start %s -> land %s;  13 poses, first t %s, last t %s m" % (
          plan["rows"][0]["foot"], plan["rows"][-1]["foot"], np.round(m["numbers"][0], 3).tolist(), np.round(m["numbers"][-1], 3).tolist()))

# THE CONTROL: the model's own shot, its own poses, read by its own code (load_sceneposes / init_cam_poses / get_pose_embeds)
vn = "D_loc2_17_t1n8_0011_Hemi12_1"
fi, lat = N.training_frames()
Pc = G.poses_in_cam(root, vn)[:, lat]
emb = np.transpose(Pc[:, :, :3, :], (0, 1, 3, 2)).reshape(Pc.shape[0], N.LATENTS, 12)
robot = "a disaster rescue robot with reinforced limbs, advanced AI, and a rugged body designed to navigate"
shots.insert(0, dict(id="control", video_name=vn, entity_prompts=[robot], loc_prompt="sunset beach",
                     prompt=robot + " is moving in the sunset beach", pose_embeds=np.round(emb, 6).tolist(), seed=24))
os.makedirs(os.path.join(TEST, "control"), exist_ok=True)
T49 = G.poses_in_cam(root, vn)[0, fi, :3, 3]
json.dump(dict(video_name=vn, rows=[dict(k=k, foot=np.round(N.lens(N.to_camera(T49[k], con)), 2).tolist(),
                                                   Z=round(float(N.to_camera(T49[k], con)[2] * 100), 2)) for k in range(N.FRAMES)],
               answer=dict(take_off=0, air_frames=N.FRAMES - 1), note="the model's own numbers through the contract"),
          open(os.path.join(TEST, "control", "plan.json"), "w"), indent=1)

inputs = dict(model=dict(code="github.com/KwaiVGI/3DTrajMaster @ 0af2932ffe514f0842bce3a2ef8809773661f8ee",
                         weights="huggingface.co/KlingTeam/3DTrajMaster @ d38a0c8fe2bdcf0ae0dfc98d1fff3bd0be5bd102 (cogvideox-5b, injector, lora)",
                         settings=SETTINGS),
              pose_layout="per entity, 13 x 12: [R | t] column by column (r00 r10 r20 r01 r11 r21 r02 r12 r22 tx ty tz), metres, "
                          "the model's camera frame (x right, y up, z forward)",
              shots=shots)
json.dump(inputs, open(os.path.join(TEST, "inputs.json"), "w"), indent=1)

rules = f"""# The rules, fixed before generation

Three shots, one pass each, seed 24, the README's inference settings ({SETTINGS['num_inference_steps']} steps, LoRA
{SETTINGS['lora_scale']}, annealed steps {SETTINGS['annealed_sample_step']}, guidance {SETTINGS['guidance_scale']}, 49 frames,
720 x 480).  The words are the model's own template, `<entity> is moving in the <location>`. Nothing in the words says
stop, jump, when or where; that information is only in the numbers.

**Measured** by `math3d/am_numbers.score`. Apple Vision subject masks run on every frame of the model's video. On each
frame, the subject nearest AM's pixel is the entity. Its error is the distance from its (centroid x, lowest row) to AM's
pixel, which comes from AM's numbers through the model's contract (`data/contract.json`, fitted on the model's own
training renders: 10.7 px median there).

| rule | passes when |
|---|---|
| FOLLOWS | median error over the frames the entity is seen <= {rule_px} px |
| ARRIVES | median error from AM's landing frame to the end <= {rule_px} px |
| ON TIME | the entity's frame-to-frame speed rises and falls with AM's: Pearson r >= 0.6 |
| NOT MEASURED | the entity is seen on fewer than 60% of the frames |

A shot PASSES only if all three rules pass.

{rule_px} px is the model's own bar: the 90th percentile of the median errors of its ten published one-entity outputs,
scored the same way (`data/baseline.json`; their median is 29.0 px). PASS means AM's numbers are followed as well as
the model follows its own.

**Control.** The model's own shot `{vn}` has its own poses and its own words. RUNNER OK means its median error is
<= {rule_px} px. The control checks the runner; it is not a verdict on AM.

**Out of the training data.** The jump lifts the entity off the ground. No training entity ever leaves it: the ground
plane fits 2,940 training feet to {ground['rms_cm']} cm rms. The slide's top speed (2.7 m/s) is inside the training
speeds (p90 {ground['speed_m_s']['p90']} m/s). The jump's (3.1 m/s across, plus the arc) is at the edge.
"""
open(os.path.join(TEST, "RULES.md"), "w").write(rules)

if os.environ.get("DRY"):
    sys.exit("dry run: nothing sealed")
lock_files = ["inputs.json", "RULES.md", "data/contract.json", "data/baseline.json", "data/ground.json", "slide/plan.json",
              "jump/plan.json", "control/plan.json", "../../math3d/am_numbers.py", "../../math3d/am_intent.py", "run/run_trajmaster.py"]
h = hashlib.sha256()
per = {}
for f in lock_files:
    b = open(os.path.join(TEST, f), "rb").read()
    per[f] = hashlib.sha256(b).hexdigest(); h.update(per[f].encode())
lock = dict(sealed_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), sha256=h.hexdigest(), files=per,
            note="written before any video is generated; the scores are held to RULES.md and nothing here is rewritten")
json.dump(lock, open(os.path.join(TEST, "lock.json"), "w"), indent=1)
open(os.path.join(TEST, "plan.sha256"), "w").write(h.hexdigest() + "\n")
print("LOCK", h.hexdigest(), "at", lock["sealed_at"])
