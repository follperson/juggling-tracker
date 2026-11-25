# Research plan for counting a practice session

Can a juggler trust the reported streak, tell where it ended, and correct a mistake by replaying the recording? Start with normal-speed phone video of a solo three-ball cascade. Preserve separate evidence for other ball counts, patterns and camera setups as they enter the study. The unit of generalization is a new practice session.

## What led to this plan

| Observation | Decision | Evidence |
|---|---|---|
| The extraction candidate matched 498 of 501 trusted reference flights. Across the full reference set it still produced 14 duplicate or unmatched catches against a ceiling of 10. | Retain the recall and overcount gates together. Use the reference corpus to detect regressions while gathering independent practice-session labels. | [Extraction candidate](https://github.com/follperson/juggling-tracker/pull/47), [flight reference](https://github.com/follperson/juggling-tracker/pull/46) |
| Two independent model-assisted visual reads found 14 definite catches and one boundary catch in a five-second indoor window. The saved input yielded zero counted catches. A held ball and an airborne ball both look like successful detections in a raw box total. | Measure airborne visibility before changing event thresholds. Keep the exposed recording in the diagnostic set and preserve its uncertainty. | [Indoor detection issue](https://github.com/follperson/juggling-tracker/issues/51), [reviewed-frame protocol](benchmarking.md#reviewed-frames-from-practice-recordings) |
| A moving held ball can produce a phantom catch, and a low final catch can trigger a drop. | Include deliberate collections, pickups and motion with no juggling as controls in every end-to-end comparison. | [Held-ball case](https://github.com/follperson/juggling-tracker/issues/48), [drop evidence](https://github.com/follperson/juggling-tracker/issues/40) |

### First controlled experiment

On that indoor recording, a model-assisted reviewer inspected six preselected, evenly spaced raw frames before opening their predictions. It marked ten definite airborne balls, six held balls and two ambiguous centers. The count came from separate temporal reviews. All current annotations await human sign-off. The reviewer froze the centers and a 60-display-pixel radius before scoring. Two full-video passes used identical local v3 weights, confidence 0.05, NMS IoU 0.7, stride one and one CPU thread. The historical saved detections have incomplete model provenance and are scored separately.

| Detector input size | Boxes on full recording | Airborne centers matched | Held centers matched | Judged positives / judged predictions | Counted catches in window | Decode + inference |
|---|---:|---:|---:|---:|---:|---:|
| 640 | 878 | 0/10 | 3/6 | 3/4 | 0 | 22.8 s |
| 1280 | 2,090 | 0/10 | 0/6 | 0/6 | 0 | 51.6 s |

Both passes processed all 1,102 frames. At 1280 the extraction candidate found two arcs over the recording and formed zero runs. At 640 it found zero arcs. The saved historical detections also matched zero of ten airborne centers at the declared radius. A 90-pixel radius credited one historical box on the neck near an airborne ball. Keep a visual error check alongside the scalar score. Each timing is a single pass, including decode, on the same machine. The six frames come from one camera and one short interval.

**Decision.** Keep the default input size. Doubling it cost time and failed to recover a catch in the reviewed window. The next diagnostic is a densely hand-labeled development sequence passed through the same event pipeline. It can distinguish inadequate detection continuity from extraction failure even with good centers. The source recording, per-frame labels, model digest and outputs are in the owner's private experiment record. The command and a synthetic end-to-end test are available in the repository. [Reproduction protocol](benchmarking.md#reviewed-frames-from-practice-recordings)

### Check against a complete practice recording

Choose the shortest of the other available normal-speed recordings by duration before reading its frames. Two model-assisted reviews agreed on 14 catches in the regular cascade, including the final collection. All 14 contact brackets overlapped between readers. The first reviewer had seen an earlier whole-clip tracker summary. The second counted independently from raw video. A later behind-the-back movement obscured possible receptions, leaving the whole-clip count at 14–17 pending human adjudication.

On the same 876 saved detections, main at `c772c28` counted four catches in one run and the extraction candidate at `5d08359` counted seven in one run. Both reported zero drops. Their run boundaries overlapped the reviewers' representative interval by 0.924 and 0.923 temporal IoU respectively. The candidate found the run and understated its count by half. This is why the recorded-review milestone tests both streak length and boundaries, and asks the juggler to settle ambiguous endings. [Candidate change](https://github.com/follperson/juggling-tracker/pull/47)

## Collect and label

1. **Start with recordings already available.** Register source hash, acquisition session, performer alias, setup, device/orientation, displayed dimensions, source clock, ball count, pattern, framing, lighting and permission for research, training and publication separately. Leave unknowns explicit. Group crops, re-encodes, near duplicates and clips from one take together. Existing clips whose results influenced an experiment remain diagnostics. [Previously observed split leakage](https://github.com/follperson/juggling-tracker/issues/38)
2. **Collect a development pilot.** Proposed initial quota is twelve independent sessions from at least four performers. Include head-to-toe and tight framing, bright and dim light, portrait and landscape, with variation across people within those conditions. In each setup record an uninterrupted run, deliberate collection, a miss and pickup, and a short period of handling balls without juggling. Keep complete takes and record natural mistakes separately from staged misses. Review consent before adding any footage to a training or public set.
3. **Freeze a separate evaluation cohort.** Recruit fresh performances and new performers before selecting a detector or thresholds. Assign whole sessions and performers to training, development or final evaluation. Publish a dated, versioned split manifest containing opaque IDs. Keep the source-to-ID key private. Once a final cohort drives a design change, record the exposure and collect a replacement for the next release claim.
4. **Label what the person did.** On raw video, independent readers mark a launch, an airborne reception, a deliberate collection, an uncontrolled miss and a pickup. Count one catch for each airborne ball visibly brought under control, including the final collection. A hand-to-hand transfer without flight contributes zero. Start a run at its first launch and bracket its last catch or first uncontrolled miss. Record individual source-frame and presentation-time brackets for contacts, including uncertain or off-camera events. Reconcile readers after both have submitted. Export established runs and drops to the existing event label format. Keep unresolved alternatives in the evidence and report them as intervals until adjudicated. [Event label contract](../src/juggletrack/eval/labels.py)
5. **Label where the balls were.** Select frame indices or timed windows before inspecting predictions. Review the entire displayed image and record all visible centers, including held centers and ambiguous contacts. Explicitly mark reviewed empty frames. Separately label short *consecutive* windows with identities through flight, contact and release to test continuity. Record blur, cropping and interventions at the frame or session level. Fit neither model nor matching radius on the evaluation split.

## Ball registration

A recorded-video pilot tested whether an explicit sample of ball color and apparent
size could recover missed flights. Two setup clips supplied one green and one red
sample. The detector used the same profiles on separate recordings from that
capture group. A proposed region needed local color contrast, a compatible size
and coherent motion across three consecutive source frames. Each input stream
went through the same event analyzer. These are development measurements with
model-assisted review and pending human sign-off.

| Recorded check | Saved detector | Registered candidates |
|---|---:|---:|
| Close-up, visible airborne centers matched across six reviewed frames (60 px) | 0/10 | 10/10 |
| Same close-up, timed catches matched in five seconds | 0/14 | 10/14 |
| Additional active window reviewed before scoring, timed catches matched in six seconds | 8/17 | 5/17 |
| Additional dense sequence, visible airborne centers matched across 240 frames (40 px) | 225/363 | 135/363 |

The close-up replay produced one extra catch in its scored window. In a separate
one-second control the juggler walked with the balls in hand. Both inputs yielded
zero catches and zero drops there. That control measures brief exposure to held
props. It gives no estimate of sensitivity to genuine drops.

We also simulated registration from the first two seconds of six Meschke clips.
We scored the next ten seconds, or through the end of shorter clips, using
one-to-one matches within 15 source pixels. Across 5,097 labeled centers the
saved detector matched 5,027 and emitted 4,761 unmatched boxes. The registered
stream matched 2,411 and emitted 791 unmatched boxes. It missed more centers on every
clip, including a low-saturation ball for which enrollment failed. Keep color
proposals available for separate visual review. Combining their detections with
the default counter needs prospective evidence on new performers, camera
distances, clothing motion and genuine drops.

### Synthetic data for detector research

In a paired test on 72 scenes, pasting a real prop texture into the same circular
mask as a rendered sphere increased box matches at intersection-over-union of 0.3
from 10 to 15. At 12 pixels diameter, each arm localized 17 of 18 centers but
matched zero boxes at that threshold. Box extent and center localization answer
different questions for a counter that fits trajectories to centers.

Before investing in synthetic training, collect real center and box labels across
ball sizes and several capture groups. Freeze a test group, then compare real-only
training with real-plus-synthetic training at the same update budget. Score center
localization, false positives and resulting catch times on those real recordings.

## Experiments and decisions

| Order | Comparison on development data | Decision it enables |
|---|---|---|
| 1. Locate the lost catches | Replay saved detections and dense human centers on the same continuously labeled windows with identical clock, surrounding context and analysis configuration. Compare visible airborne recall, longest gaps in a flight, recovered arcs and catch timestamps. Evaluate the interior so clipped boundary flights cannot masquerade as misses. | If human centers recover flights, direct effort to detection or continuity. If they also fail, inspect extraction, hand height and run formation before buying more labels. |
| 2. Fix the evidenced stage | For a detector limitation, compare a fixed baseline to one crop or training-data intervention at a time on separate recordings. Include full-image coordinates, held balls, clutter and true negatives. For extraction failures, replay exactly the same detections and compare changes per session against the existing controls. | Keep a candidate only if it improves the intended failure across independent sessions and retains the existing overcount, false-drop and reference-recall gates. Report regressions individually. |
| 3. Test the actual practice workflow | Have a juggler import an ordinary recording, find the longest run, replay a disputed ending and correct it without editing JSON. Record task completion, correction time and whether the corrected summary survives re-opening. | Advance the recorded review when a participant can locate and fix a known error. Use observed correction effort to choose the next interface change. |
| 4. Verify live and broaden scope | Run complete recorded-clock streams, then device-camera sessions with the same event definitions. Evaluate four-ball fountain and a mixed-height three-ball pattern as distinct cohorts before claiming either. Include cross-performer and cross-camera tests within each. | Name supported cases only after that cohort meets the product gates. A simulator run can qualify a regression test. A release claim needs recorded human performance. |

## Reporting and release evidence

Report the number of performers, sessions, labeled runs, true drops and minutes of non-juggling exposure. Use all labeled runs in the catch-within-one and boundary-overlap denominators, including unmatched runs. Report extra runs, drop TP/FP/FN, false catches per non-juggling minute, and catch errors per run. Compute both pooled and session-level results. Show condition-level denominators. For cross-performer uncertainty, resample performers first and sessions within each sampled performer when there are enough independent performers. For pilot data, publish raw fractions and individual failures. For live use include frame throughput, event confirmation delay and duplicate counts on full sessions.

Before a candidate sees the final dataset, freeze its commit, weights, decoder, settings, label version, split and release criteria. The [product roadmap](product-roadmap.md) gives the proposed gates. For every experiment preserve the question, preselected comparison, exact command, input and implementation digests, result, decision and reason to revisit. Archive predictions separately from user corrections. A future reviewer should be able to explain a changed count at its time in the source video.
