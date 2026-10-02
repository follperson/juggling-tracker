# Detector diagnostics — September 7, 2026

**Decision:** keep detector defaults unchanged. Confidence tuning helps one
failure case but sharply reduces recall on another. Higher input resolution
adds false positives on the long clip whose main failure is event counting.
Prioritize reviewed training examples from low-confidence/duplicate-heavy
footage and persistent live event identities.

## Method

Compared existing saved v3 detections to the Meschke dataset's manually labeled
ball centers, without arc extraction or generated event labels. Used the new
`detection-eval` command with a 10-pixel matching tolerance, stride 1, and
one-to-one assignment. The confidence sweep filters the same saved detections;
it does not rerun the model. All of these clips have been used in development.
This is not an untouched test set, and these are center metrics, not box mAP.

Citation: Stephen Meschke — Juggling Data Set —
https://sites.google.com/view/jugglingdataset

## Confidence comparison

| Video | Minimum confidence | Center precision | Center recall | False positives | Missed balls |
|---|---:|---:|---:|---:|---:|
| ss3_id_016 | 0.05 | 99.6% | 100.0% | 66 | 0 |
| ss3_id_016 | 0.10 | 99.9% | 100.0% | 13 | 0 |
| ss3_id_016 | 0.25 | 100.0% | 100.0% | 1 | 0 |
| ss3_id_016 | 0.50 | 100.0% | 100.0% | 0 | 0 |
| ss3_id_110 | 0.05 | 56.3% | 89.9% | 549 | 79 |
| ss3_id_110 | 0.10 | 70.1% | 88.0% | 295 | 94 |
| ss3_id_110 | 0.25 | 85.3% | 85.8% | 116 | 112 |
| ss3_id_110 | 0.50 | 89.9% | 80.5% | 71 | 153 |
| ss441_id_089 | 0.05 | 13.5% | 76.3% | 5,186 | 251 |
| ss441_id_089 | 0.10 | 22.6% | 63.1% | 2,282 | 390 |
| ss441_id_089 | 0.25 | 50.5% | 23.5% | 244 | 809 |
| ss441_id_089 | 0.50 | n/a | 0.0% | 0 | 1,058 |

At confidence 0.05, ss3_id_016 has 18,450 true matches and a mean matched
center error of 0.81 pixels. Its poor event counts cannot be explained by
low detector recall under this metric.

At confidence 0.05, ss3_id_110 has 189 duplicate candidates and 360 other
false positives; ss441_id_089 has 1,498 and 3,688 respectively. The latter
contains 354 scored frames; the CSV's extra trailing frame is excluded.
Other false positives include localization misses as well as background
objects, so visual inspection is needed to classify their training value.

## Resolution comparison

| ss3_id_016 saved input size | Center precision | Center recall | Duplicate candidates | Other false positives |
|---|---:|---:|---:|---:|
| 640 | 99.6% | 100.0% | 58 | 8 |
| 1280 | 90.7% | 100.0% | 949 | 946 |

The 1280 detections come from the separately saved
`outputs/meschke-val/post-hardening/ss3_id_016-1280/detections.jsonl` experiment.
They produce 1,895 false positives versus 66 at 640, without recovering an
additional labeled ball at this tolerance. These historical files lack a
complete modern inference manifest; input size/model identity follow the
recorded experiment, not a fresh controlled detector run.

## Next experiments

1. Visually review error frames from ss3_id_110 and ss441_id_089, separating
   duplicate boxes, loose localization, background objects, and missing balls.
2. Collect new recordings with the same challenging conditions. Hold out
   entire recordings/people/locations before selecting thresholds or training.
3. Compare a person/flight-region crop and reviewed fine-tuning data against
   the current model. Avoid choosing a global confidence threshold from the
   best-looking single clip.
4. Keep long-session event reconstruction as a separate priority. The current
   direct realtime replay still counts 210 catches against offline's 41 on
   ss3_id_016 despite its strong detector-center results.

Local detailed outputs: `outputs/accuracy-review-2026-09-07/` contains the three
640 center reports, the 1280 report, and `confidence-sweep.json`. The individual
reports include detection/CSV hashes; the commands and matching behavior are
documented in [benchmarking.md](benchmarking.md). These outputs are ignored by
Git; this document preserves the measured summary.
