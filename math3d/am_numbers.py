"""AM'S NUMBERS STRAIGHT INTO A NUMBER-TAKING MODEL, no guide picture.  3DTrajMaster (ICLR 2025; its open release is
CogVideoX-5B with a LoRA and a pose injector) makes a video from words plus, for each entity, 13 poses [R | t] in its
camera's frame (x right, y up, z forward, metres): the numbers ARE the input.  AM writes those numbers itself.
  THE MODEL'S CONTRACT: the frame the poses are written in is not the camera that rendered the model's training videos.
    It is fitted once, on the model's own training renders (camera 12, one entity, Apple Vision subject masks): the
    subject's (mask-centroid x, lowest row) = lens(R^T (t - c)), f = 1060.606 px at 720 x 480 (a 35 mm lens on a
    23.76 mm sensor).  x = R^T (t - c) is the subject's foot in the camera that really rendered the clip.
  AM'S WORLD: the ground the training entities stand on, fitted in that camera, is AM's stage; the entity is a piece on
    it; AM solves the intent there (am_intent.solve): a slide, e = 3 s^2 - 2 s^3, starting and stopping at rest, or a
    jump whose every airborne frame has the same second difference, gravity.  The model's 49 frames are 2.8 s of its
    training clips (frames 10..94 of 99 at 30 fps, stride 1.75), so AM works at 30 / 1.75 = 17.14 frames a second.
  TO THE MODEL: each AM frame goes back to the poses' frame, t = R x + c, facing the way the piece travels and upright
    as every training entity is; the 13 poses are AM's frames at exactly the moments the model was trained on (the
    first frame, then the middle of each 4-frame latent), laid out as the model reads them.  Nothing is drawn.
  THE SCORE: Apple Vision subject masks on every frame of the model's video; on each frame the subject nearest AM's
    pixel is the entity, and its distance from AM's (centroid x, lowest row) is the error.  The same eyes and the same
    rule, on the model's own 30 published outputs, give its baseline.
    world(out, con, ground, start, size)          -> out/map.json, out/puzzle.json
    to_model(plan, con)                           -> the 13 poses, and AM's numbers in the poses' frame for all 49 frames
    score(video, plan, rule_px, out)              -> out/score.json, out/labelled.mp4"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EYES = os.path.join(HERE, "video_instances")
F_PX, CX, CY, W, H = 35.0 / 23.76 * 720.0, 360.0, 240.0, 720, 480
STRIDE, FIRST, FRAMES, LATENTS = 1.75, 10, 49, 13
FPS = 30.0 / STRIDE
UP = np.array([0.0, 0.9956342260592881, -0.09334070869305827])      # every training entity's up axis, poses' frame


def training_frames():
    """the 99-frame clip's frames the model learned to make, and the 13 it was given poses at (its own code)"""
    fi = np.linspace(FIRST, FIRST + int(FRAMES * STRIDE) - 1, FRAMES, dtype=int)
    lat = np.zeros_like(fi[::4])
    lat[0] = fi[0]
    lat[1:] = ((fi[1:][::4] + fi[4:][::4]) / 2).astype(np.int64)
    return fi, lat


def contract(path):
    c = json.load(open(path))
    return dict(R=np.array(c["R"]), c=np.array(c["c"]), raw=c)


def lens(x):
    """camera points (x right, y up, z forward) -> pixels, y down"""
    x = np.asarray(x, float)
    return np.stack([CX + F_PX * x[..., 0] / x[..., 2], CY - F_PX * x[..., 1] / x[..., 2]], -1)


def to_camera(t, con):
    return (np.asarray(t, float) - con["c"]) @ con["R"]


def to_poses(x, con):
    return np.asarray(x, float) @ con["R"].T + con["c"]


def stage(ground):
    """the fitted ground y = a x + b z + d (camera, metres) as AM's stage (cm, y down, normal pointing up)"""
    a, b, d = ground["a"], ground["b"], ground["d"]
    n = np.array([-a, 1.0, -b]); n /= np.linalg.norm(n)
    n = np.array([n[0], -n[1], n[2]])                                  # AM's frame has y down; this normal points up
    O = np.array([0.0, -(b * 8.0 + d), 8.0]) * 100.0                   # the ground straight ahead, 8 m out
    e1 = np.array([1.0, 0, 0]) - n[0] * n; e1 /= np.linalg.norm(e1)
    e2 = np.array([0, 0, 1.0]) - n[2] * n - (np.array([0, 0, 1.0]) @ e1) * e1; e2 /= np.linalg.norm(e2)
    return dict(normal=n.round(6).tolist(), offset=round(float(-n @ O), 4),
                axes=dict(across=e1.round(6).tolist(), forward=e2.round(6).tolist(), origin=O.round(3).tolist()))


def world(out, con, ground, start, size, frame_seconds=FRAMES / FPS):
    """AM's puzzle for the model's world: its lens, its ground as the stage, one piece (the entity) standing at start
    = (across, forward) cm; size = (across, forward_seen, up) cm."""
    os.makedirs(out, exist_ok=True)
    st = stage(ground)
    O, e1, e2 = (np.array(st["axes"][k]) for k in ("origin", "across", "forward"))
    foot = O + start[0] * e1 + start[1] * e2
    Z = foot[2]
    w, h = F_PX * size[0] / Z, F_PX * size[2] / Z
    fx, fy = CX + F_PX * foot[0] / Z, CY + F_PX * foot[1] / Z
    piece = dict(id=1, kind="subject", stands=dict(across=float(start[0]), forward=float(start[1]), up=0.0),
                 size=dict(across=float(size[0]), forward_seen=float(size[1]), up=float(size[2])), top=float(size[2]),
                 rests_on=None, distance_cm=round(float(np.linalg.norm(foot)), 1),
                 box=[round(fx - w / 2, 1), round(fy - h, 1), round(w, 1), round(h, 1)])
    pz = dict(clip=None, frame=0, size=[W, H], lens=dict(f=F_PX, cx=CX, cy=CY, fov=round(float(np.degrees(2 * np.arctan(CX / F_PX))), 3)),
              stage=st, pieces=[piece], relations=[], source="the camera that rendered 3DTrajMaster's training clips (fitted)")
    json.dump(pz, open(os.path.join(out, "puzzle.json"), "w"), indent=1)
    json.dump(dict(fps=FPS, frames=FRAMES, seconds=round(frame_seconds, 4)), open(os.path.join(out, "map.json"), "w"), indent=1)
    return pz


def to_model(plan, con):
    rows = plan["rows"]
    am = np.array([[r["X"], -r["Y"], r["Z"]] for r in rows]) / 100.0     # AM's foot, camera frame, metres, y up
    t = to_poses(am, con)                                                # the same points in the poses' frame
    way = t[-1] - t[0]
    way = way - (way @ UP) * UP
    fwd = way / np.linalg.norm(way)                                       # facing the way it travels, upright
    right = np.cross(UP, fwd)
    Rent = np.stack([right, UP, fwd], 1)
    fi, lat = training_frames()
    k = (lat - FIRST) / STRIDE                                            # those moments in AM's frames
    tl = np.stack([np.interp(k, np.arange(len(t)), t[:, i]) for i in range(3)], 1)
    P = np.concatenate([np.repeat(Rent[None], LATENTS, 0), tl[:, :, None]], 2)   # (13, 3, 4) = [R | t]
    emb = np.transpose(P, (0, 2, 1)).reshape(1, LATENTS, 12)                     # the model's layout: column by column
    return dict(pose_embeds=emb, latent_am_frames=k.tolist(), latent_clip_frames=lat.tolist(), numbers=t, facing=Rent)


def expected(plan):
    return np.array([r["foot"] for r in plan["rows"]], float)


def eyes(video, out_jsonl):
    with open(out_jsonl, "w") as f:
        subprocess.run([EYES, video], stdout=f, check=True)
    return [json.loads(l) for l in open(out_jsonl)]


def follow(frames, E):
    """per frame: the subject nearest AM's pixel; its (centroid x, lowest row), or nan where nothing is seen"""
    obs = np.full((len(E), 2), np.nan)
    for fr in frames:
        k = fr["t"]
        if k >= len(E):
            break
        ok = [i for i in fr["inst"] if i[0] > 1 and i[2] < W - 2 and i[3] < H - 2 and i[4] >= 150]
        if ok:
            F = np.array([[i[5], i[3]] for i in ok])
            obs[k] = F[np.argmin(np.linalg.norm(F - E[k], axis=1))]
    return obs


def score(video, plan, rule_px, out, seen_min=0.6, speed_r=0.6):
    """the rules, fixed before generation: FOLLOWS (median error over the frames the entity is seen <= rule_px),
    ARRIVES (median error from AM's landing frame on <= rule_px), ON TIME (the entity's speed, frame to frame, rises and
    falls with AM's: Pearson r >= speed_r); seen on fewer than seen_min of the frames: NOT MEASURED"""
    os.makedirs(out, exist_ok=True)
    E = expected(plan)
    fr = eyes(video, os.path.join(out, "eyes.jsonl"))
    obs = follow(fr, E)
    err = np.linalg.norm(obs - E, axis=1)
    seen = ~np.isnan(err)
    land = plan["answer"]["take_off"] + plan["answer"]["air_frames"]
    sm = lambda a: np.array([np.nanmean(a[max(0, i - 1):i + 2], 0) for i in range(len(a))])
    vo = np.linalg.norm(np.diff(sm(obs), axis=0), axis=1)
    ve = np.linalg.norm(np.diff(E, axis=0), axis=1)
    ok = ~np.isnan(vo)
    r = float(np.corrcoef(vo[ok], ve[ok])[0, 1]) if ok.sum() > 3 else float("nan")
    rows = plan["rows"]
    Z = np.array([x["Z"] for x in rows]) / 100.0
    s = dict(video=os.path.basename(video), frames=len(E), seen=int(seen.sum()), seen_share=round(float(seen.mean()), 3),
             median_px=round(float(np.nanmedian(err)), 1) if seen.any() else None,
             p90_px=round(float(np.nanpercentile(err, 90)), 1) if seen.any() else None,
             median_cm=round(float(np.nanmedian(err * Z / F_PX * 100)), 1) if seen.any() else None,
             landed_from=int(land), landed_median_px=round(float(np.nanmedian(err[land:])), 1) if seen[land:].any() else None,
             speed_r=round(r, 3), rule_px=rule_px, speed_r_min=speed_r, seen_min=seen_min,
             per_frame=[dict(k=k, am=[round(float(v), 1) for v in E[k]],
                             seen=None if np.isnan(obs[k, 0]) else [round(float(v), 1) for v in obs[k]],
                             err_px=None if np.isnan(err[k]) else round(float(err[k]), 1)) for k in range(len(E))])
    if s["seen_share"] < seen_min:
        s.update(verdict="NOT MEASURED", why=["the entity is seen on %d of %d frames" % (s["seen"], len(E))])
    else:
        rules = dict(FOLLOWS=s["median_px"] <= rule_px,
                     ARRIVES=s["landed_median_px"] is not None and s["landed_median_px"] <= rule_px,
                     ON_TIME=bool(r >= speed_r))
        s.update(rules=rules, verdict="PASS" if all(rules.values()) else "FAIL",
                 why=[k for k, v in rules.items() if not v])
    json.dump(s, open(os.path.join(out, "score.json"), "w"), indent=1)
    label(video, E, obs, s, os.path.join(out, "labelled.mp4"))
    return s


def label(video, E, obs, s, out_mp4):
    """the model's frames with AM's pixel (cross) and the entity's point (ring) on each"""
    from PIL import Image, ImageDraw
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    F = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                          "-r", "%.6f" % FPS, "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", out_mp4],
                         stdin=subprocess.PIPE)
    for k in range(min(len(F), len(E))):
        im = Image.fromarray(F[k].copy()); d = ImageDraw.Draw(im)
        for j in range(len(E) - 1):
            d.line([tuple(E[j]), tuple(E[j + 1])], fill=(255, 255, 255), width=1)
        x, y = E[k]
        d.line([x - 9, y, x + 9, y], fill=(255, 40, 40), width=3); d.line([x, y - 9, x, y + 9], fill=(255, 40, 40), width=3)
        if not np.isnan(obs[k, 0]):
            ox, oy = obs[k]
            d.ellipse([ox - 7, oy - 7, ox + 7, oy + 7], outline=(40, 220, 255), width=3)
            d.text((8, 8), "frame %d  AM (red +) vs the entity (blue o): %.0f px" % (k, np.hypot(ox - x, oy - y)), fill=(255, 255, 255))
        else:
            d.text((8, 8), "frame %d  entity not seen" % k, fill=(255, 255, 255))
        d.text((8, 22), "%s  median %s px" % (s.get("verdict", ""), s.get("median_px")), fill=(255, 255, 255))
        p.stdin.write(np.asarray(im).tobytes())
    p.stdin.close(); p.wait()
