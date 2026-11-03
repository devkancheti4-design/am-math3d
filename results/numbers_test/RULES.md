# The rules, fixed before generation

Three shots, one pass each, seed 24, the README's inference settings (50 steps, LoRA
0.6, annealed steps 20, guidance 6.0, 49 frames,
720 x 480).  The words are the model's own template, `<entity> is moving in the <location>`. Nothing in the words says
stop, jump, when or where; that information is only in the numbers.

**Measured** by `math3d/am_numbers.score`. Apple Vision subject masks run on every frame of the model's video. On each
frame, the subject nearest AM's pixel is the entity. Its error is the distance from its (centroid x, lowest row) to AM's
pixel, which comes from AM's numbers through the model's contract (`data/contract.json`, fitted on the model's own
training renders: 10.7 px median there).

| rule | passes when |
|---|---|
| FOLLOWS | median error over the frames the entity is seen <= 44.7 px |
| ARRIVES | median error from AM's landing frame to the end <= 44.7 px |
| ON TIME | the entity's frame-to-frame speed rises and falls with AM's: Pearson r >= 0.6 |
| NOT MEASURED | the entity is seen on fewer than 60% of the frames |

A shot PASSES only if all three rules pass.

44.7 px is the model's own bar: the 90th percentile of the median errors of its ten published one-entity outputs,
scored the same way (`data/baseline.json`; their median is 29.0 px). PASS means AM's numbers are followed as well as
the model follows its own.

**Control.** The model's own shot `D_loc2_17_t1n8_0011_Hemi12_1` has its own poses and its own words. RUNNER OK means its median error is
<= 44.7 px. The control checks the runner; it is not a verdict on AM.

**Out of the training data.** The jump lifts the entity off the ground. No training entity ever leaves it: the ground
plane fits 2,940 training feet to 10.47 cm rms. The slide's top speed (2.7 m/s) is inside the training
speeds (p90 3.08 m/s). The jump's (3.1 m/s across, plus the arc) is at the edge.
