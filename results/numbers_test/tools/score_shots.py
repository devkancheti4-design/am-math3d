"""As run: the score of every video that came back, held to RULES.md.  First the lock is checked: every sealed file
must still have the hash it had before generation, or nothing is scored.
    python3 score_shots.py            -> out/<shot>/score.json, out/<shot>/labelled.mp4, out/verdicts.json"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TEST, "..", "..", "math3d"))
import am_numbers as N                                          # noqa: E402

lock = json.load(open(os.path.join(TEST, "lock.json")))
h = hashlib.sha256()
for f, want in lock["files"].items():
    got = hashlib.sha256(open(os.path.join(TEST, f), "rb").read()).hexdigest()
    assert got == want, "LOCK BROKEN: %s changed after sealing" % f
    h.update(got.encode())
assert h.hexdigest() == lock["sha256"], "LOCK BROKEN"
print("lock holds:", lock["sha256"], "sealed", lock["sealed_at"])

rule_px = json.load(open(os.path.join(TEST, "data", "baseline.json")))["summary"]["rule_px"]
out = os.path.join(TEST, "out")
os.makedirs(out, exist_ok=True)
verdicts = dict(lock=lock["sha256"], rule_px=rule_px, shots={})
for sid in ("control", "slide", "jump"):
    video = os.path.join(out, sid + ".mp4")
    if not os.path.exists(video):
        continue
    plan = json.load(open(os.path.join(TEST, sid, "plan.json")))
    s = N.score(video, plan, rule_px, os.path.join(out, sid))
    v = dict(verdict=s["verdict"], why=s["why"], median_px=s["median_px"], p90_px=s["p90_px"], median_cm=s["median_cm"],
             landed_median_px=s["landed_median_px"], speed_r=s["speed_r"], seen_share=s["seen_share"])
    if sid == "control":
        v["verdict"] = "RUNNER OK" if s["median_px"] is not None and s["median_px"] <= rule_px else "RUNNER NOT OK"
    verdicts["shots"][sid] = v
    print(sid, json.dumps(v))
json.dump(verdicts, open(os.path.join(out, "verdicts.json"), "w"), indent=1)
