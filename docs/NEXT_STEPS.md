# Next steps using public data

The pitch remains the goal: **camera → crystal measurements and alerts → PID plus
an adaptive supervisor → operator review of proposed actions**. Public datasets
can support model development now. Real control and improved crystal quality need
different evidence that the current files do not provide.

The software foundation runs: synthetic training, video observation, candidate
alerts, temperature PID and the operator replay. The next core deliverable is a
small, measured real-image model comparison. More simulation alone will not
establish real detection accuracy.

1. **Check the small public KDP pack.** We already have 881 indexed frames and six
   native annotation sheets. An outline is the example answer telling the model
   where the crystal ends. A person checks that answer against the image; unclear
   cases stay uncertain. No real reviews have been recorded yet. Start with one
   image, then the six. The [review guide](KDP_VIDEO_DATA.md#check-one-image) gives
   the command and explains what to look for. Ambiguous defects need appropriate
   knowledge; software-generated suggestions are not independent answers.
2. **Build the development examples around what is visible.** Expand the checked
   development pack across source runs and difficult views. For events, annotate
   short consecutive periods before/after first visibility, plus negative periods.
   Preserve whole-run groups and source permissions. Seek additional public KDP
   growth data with labels and timing; the current three clips form one study
   family and cannot provide an independent evaluation cohort.
   A second [KDP gel-growth study](KDP_GEL_DATA.md) now adds nine published panel
   candidates. They retain their small resolution, source marks and separate
   domain; actual reviews and an independent evaluation cohort are still needed.
   The [CrystalCV source](CRYSTALCV_DATA.md) also adds perovskite growth images,
   published rectangle measurements and a candidate process log. Audit its
   synchronization, causal fields and source dependencies before a separate
   action-response development experiment; keep KDP labels and evaluation open.
3. **Compare models fairly.** Use compatible reviewed labels to compare
   synthetic-only, real-only and mixed YOLOv8-seg training. Freeze the evaluation
   rules and independent real groups before scoring. Report outline error, missed
   crystals, false alerts and event delay where timing supports it. Existing
   alpha-LGA shape-proxy results are useful supporting evidence, with their own
   narrower task. The current inclusion candidate remains unpromoted.
4. **Demonstrate the complete software path.** In the existing replay console,
   show the source video, measured outlines, localized evidence, forecast and
   proposed operator action. Retain PID/rule/recent-rate baselines. Real videos
   support shadow proposals; bounded recipe pauses currently use a simulated
   plant. A fixed video cannot prove that an action prevents a defect.
5. **Close the remaining pitch measurements.** Face/asymmetry interpretation,
   reliable visible inclusions, rotation/flow responses and a useful adaptive
   action-response objective remain open. First-hour detection needs a trusted
   growth-start clock; control benefit needs synchronized measured/applied actions
   and subsequent observations. Continue looking for suitable public evidence.
   The [edge diagnostic](PROJECTED_EDGES.md) now shows why a stable reference,
   known pose and attachment evidence are needed. Its constructed-image result
   does not supply real face labels or real accuracy.
6. **Keep the supporting experiments bounded.** The CPU/hybrid GPU flow comparison
   and the first Al-precursor VQE/classical diagnostic are recorded. A separate
   [heat/solute/buoyancy component](THERMAL_TRANSPORT.md) now passes manufactured
   numerical checks. KDP thermal calibration, physical risk maps, converged KDP
   surface chemistry and impurity thresholds remain open. These experiments
   support the pitch; they do not replace the camera/data/control work.

If no reviewer is available yet, keep the real-label gate pending and continue
work whose evidence is already available. Do not call uncertain drafts accepted
labels or claim real accuracy from synthetic scores. The detailed
[roadmap](../roadmap.md) preserves completed experiments, failures and each
remaining pitch requirement.
