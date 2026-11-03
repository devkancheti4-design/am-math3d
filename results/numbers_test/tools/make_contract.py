"""As run (3 Nov 2026): the model's contract, its baseline, and its ground, from 3DTrajMaster's own files.
    python3 make_contract.py <360Motion root> <published outputs dir> <eyes dir> <out data dir>
<360Motion root>: Hemi12_transforms.json and Desert/ (location_data.json; for each of the 30 test shots in
test_names.json, its pose file and its camera-12 render), read out of Desert.zip with rangezip.py.
<published outputs dir>: the 30 videos (and .txt) in the repo's CogVideo/inference/output_example.
  1. Apple Vision subject masks (math3d/video_instances) on every frame of the 30 renders and the 30 outputs.
  2. THE CONTRACT, fitted on the 10 one-entity renders: (centroid x, lowest row) = lens(R^T (t - c)), soft-L1 loss.
  3. THE BASELINE: every published output and every render, at the 49 frames the model makes, scored by
     am_numbers.follow against the contract's pixel for the numbers the model was given.  The rule for AM's shots is
     the 90th percentile of the ten one-entity outputs' median errors.
  4. THE GROUND: a plane through every training foot (all 30 shots) in the camera that rendered them."""
import glob, json, os, sys
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "math3d"))
import tjm_geom as G                                            # noqa: E402  (the model's own pose code)
import am_numbers as N                                          # noqa: E402

root, outs, eyedir, data = sys.argv[1:5]
names = sorted(json.load(open(os.path.join(data, "test_names.json"))))
os.makedirs(os.path.join(eyedir, "ue"), exist_ok=True); os.makedirs(os.path.join(eyedir, "out"), exist_ok=True)
published = {open(p[:-4] + ".txt").read().split("\n")[0].strip(): p for p in glob.glob(os.path.join(outs, "*.mp4"))}
ue, out = {}, {}
for vn in names:
    for d, vid, store in (("ue", os.path.join(root, "Desert", vn, "videos", vn + "_C_12_35mm.mp4"), ue), ("out", published[vn], out)):
        j = os.path.join(eyedir, d, vn + ".jsonl")
        store[vn] = [json.loads(l) for l in open(j)] if os.path.exists(j) else N.eyes(vid, j)

fi, _ = N.training_frames()
P = {vn: G.poses_in_cam(root, vn) for vn in names}
singles = [vn for vn in names if P[vn].shape[0] == 1]
T, FU = [], []
for vn in singles:
    for fr in ue[vn]:
        ok = [i for i in fr["inst"] if i[0] > 1 and i[2] < N.W - 2 and i[3] < N.H - 2 and i[4] >= 150]
        if ok:
            i = max(ok, key=lambda i: i[4])
            T.append(P[vn][0, fr["t"], :3, 3]); FU.append([i[5], i[3]])
T, FU = np.array(T), np.array(FU)
pred = lambda q, T: N.lens((T - q[3:]) @ Rot.from_rotvec(q[:3]).as_matrix())
q = optimize.least_squares(lambda q: (pred(q, T) - FU).ravel(), np.r_[0, 0, 0, 0, 3.4, 0], loss="soft_l1", f_scale=20).x
e = np.linalg.norm(pred(q, T) - FU, axis=1)
R = Rot.from_rotvec(q[:3]).as_matrix()
con = dict(R=R.tolist(), c=q[3:].tolist(), turn_deg_xyz=Rot.from_matrix(R).as_euler("xyz", degrees=True).round(3).tolist(),
           f=N.F_PX, cx=N.CX, cy=N.CY, size=[N.W, N.H], feature="Apple Vision subject mask: centroid x, lowest row",
           fitted_on=singles, frames=int(len(e)), median_px=round(float(np.median(e)), 2), p90_px=round(float(np.percentile(e, 90)), 2))
json.dump(con, open(os.path.join(data, "contract.json"), "w"), indent=1)
print("contract: the rendering camera sits at %s m in the poses' frame, turned %s deg; on its own renders: median %.1f px, "
      "p90 %.1f px (%d frames)" % (np.round(q[3:], 3), con["turn_deg_xyz"], con["median_px"], con["p90_px"], con["frames"]))

C = N.contract(os.path.join(data, "contract.json"))
base = {}
for vn in names:
    n = P[vn].shape[0]
    T49 = P[vn][:, fi][:, :, :3, 3]
    row = dict(entities=n, depth_m=round(float(np.median(T49[..., 2])), 2))
    for d, store in (("output", out), ("render", ue)):
        frames = store[vn] if d == "output" else [dict(t=k, inst=store[vn][f]["inst"]) for k, f in enumerate(fi)]
        errs = []
        for k in range(n):
            E = N.lens(N.to_camera(T49[k], C))
            errs.append(np.linalg.norm(N.follow(frames, E) - E, axis=1))
        errs = np.concatenate(errs)
        row[d] = dict(median_px=round(float(np.nanmedian(errs)), 1), seen_share=round(float(np.mean(~np.isnan(errs))), 3))
    base[vn] = row
one = [base[v]["output"]["median_px"] for v in singles]
rule = round(float(np.percentile(one, 90)), 1)
summary = dict(rule_px=rule, rule_is="90th percentile of the ten one-entity published outputs' median errors",
               one_entity_output_medians=sorted(one), one_entity_output_median_of_medians=round(float(np.median(one)), 1),
               one_entity_render_median_of_medians=round(float(np.median([base[v]["render"]["median_px"] for v in singles])), 1))
for n in (1, 2, 3):
    vs = [v for v in names if base[v]["entities"] == n]
    summary["%d_entity" % n] = dict(shots=len(vs), output_median_of_medians=round(float(np.median([base[v]["output"]["median_px"] for v in vs])), 1),
                                    render_median_of_medians=round(float(np.median([base[v]["render"]["median_px"] for v in vs])), 1))
json.dump(dict(summary=summary, shots=base), open(os.path.join(data, "baseline.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

F = np.concatenate([N.to_camera(P[vn][k][fi][:, :3, 3], C) for vn in names for k in range(P[vn].shape[0])])
A = np.c_[F[:, 0], F[:, 2], np.ones(len(F))]
co, *_ = np.linalg.lstsq(A, F[:, 1], rcond=None)
res = F[:, 1] - A @ co
gr = dict(a=float(co[0]), b=float(co[1]), d=float(co[2]), rms_cm=round(float(res.std() * 100), 2), feet=int(len(F)),
          x_m=np.percentile(F[:, 0], [1, 99]).round(2).tolist(), z_m=np.percentile(F[:, 2], [1, 99]).round(2).tolist(),
          speed_m_s=dict(median=None))
sp = [np.linalg.norm(np.diff(N.to_camera(P[vn][k][fi][:, :3, 3], C)[:, [0, 2]], axis=0), axis=1) * N.FPS
      for vn in names for k in range(P[vn].shape[0])]
sp = np.concatenate(sp)
gr["speed_m_s"] = dict(median=round(float(np.median(sp)), 2), p90=round(float(np.percentile(sp, 90)), 2), p99=round(float(np.percentile(sp, 99)), 2))
json.dump(gr, open(os.path.join(data, "ground.json"), "w"), indent=1)
print("ground: y = %.4f x + %.4f z + %.3f (m), rms %.1f cm over %d feet; x %s m, z %s m; speed %s" %
      (gr["a"], gr["b"], gr["d"], gr["rms_cm"], gr["feet"], gr["x_m"], gr["z_m"], gr["speed_m_s"]))
