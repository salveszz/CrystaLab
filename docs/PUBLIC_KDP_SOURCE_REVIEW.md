# Public KDP source review — 2026-11-12

Current access is **public datasets only**. Acquired KDP movies, Leeds images/shape
fits and CHiMP masks retain their separate [licenses and limits](REAL_VISION_DATA.md).
The completed [shape fits](SOURCE_SHAPE_TRAINING.md) measure alpha-LGA silhouette
proxy agreement, not KDP inclusion, secondary-crystal or asymmetry accuracy.

This reviews specific additional leads; it does not show that no suitable public
dataset exists. Public availability does not itself establish human review.

| Candidate | Evidence checked | Decision |
|---|---|---|
| [Cai et al., KDP gel-growth, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210260/) | Versioned PMC JSON/XML and three original figures, CC BY 4.0. Nine microscopy crops include sparse 0/10/70-minute gel-growth panels; no raw video/masks/control logs in the inspected payload. | [Acquired and independently audited](KDP_GEL_DATA.md) as a separate publication-panel development family. Nine sheets remain uncertain/unreviewed; no model adoption or real accuracy claim. |
| [He, IEEE DataPort, DOI 10.21227/fm8f-8t88](https://ieee-dataport.org/documents/images-micro-craters-and-micro-cracks-surface-kdp-crystals) | The page describes 338 static KDP surface images, 240 micro-craters and 200 micro-cracks under black-support/ring-light imaging. Instructions ask the user to create LabelImg annotations. The available page says files have not been uploaded and describes subscription access. Supplied labels and reuse terms are unverified. | Surface-inspection lead; no payload acquired. These are not in-solution growth-event labels. |
| [Feng et al., 2025, enhanced YOLOv5](https://link.springer.com/article/10.1007/s11554-025-01654-6), [author repository](https://github.com/Good-he/EXN-yolo) | The public article preview links the repository, whose README advertises a Baidu share. Anonymous HTTP returned an extraction-code form, not an archive. Repository metadata returns `license: null`; dataset reuse terms remain unverified. The preview does not resolve the payload/label/license questions. | Hold pending inspectable payload and applicable reuse terms. Keep the surface task separate. No purchase, login, author contact or training adoption. |
| [Zhu et al., 2025, DKDP laser damage](https://www.sciencedirect.com/science/article/abs/pii/S0925346725007396) | The primary abstract describes laser-damage detection. The full page request was inaccessible during this check; no public image/label payload was verified. | Related downstream-damage lead. No acquisition or growth-class mapping. |

DataPort and EXN involve overlapping authors and related surface tasks. Image
overlap and cohort independence are unknown. Do not count them as independent
evaluation sets or adopt random image splits without checking specimen/run
identity and duplicates. The EXN README conversion example does not establish
the actual payload's class map.

Anonymous access evidence is preserved locally in
`data/audits/kdp_candidate_access_20261112_v1/report.json`, including response
bytes, URLs, status, content type and SHA256. HTTP 200 for an HTML form is not a
dataset download and does not prove that login is required. No images, labels
or physical measurements were added from these leads.

## Next usable evidence

Prioritize public original KDP growth images/video through solution, with reviewed
pixel/event labels or annotations a real reviewer can check. Preserve material,
phase, acquisition clocks and run/specimen groups. Record absent event classes
and occlusions. Freeze whole-run groups before fitting or tuning.

First-hour claims need a documented growth-start clock and relevant event
coverage. Action-response claims need timestamped temperature/rotation/flow
commands and subsequent observations. Quality/yield claims need independent
measured outcomes. Sources missing these fields retain their narrower use.

Continue the annotation-review workflow on available development data and seek
better public growth/event sources. In plain terms, a real person must check
whether each proposed outline follows the crystal and whether the claimed event
is visible. The software can prepare and preserve proposals; it cannot perform
independent human review itself.

## Related other-material control-data acquisition — 2026-11-13

The [CrystalCV study](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00140h)
links public, versioned perovskite images and software/examples. Those archives
are now [acquired and indexed separately](CRYSTALCV_DATA.md), including source
rectangle measurements and a process-log lead. They are useful for reviewing
observable trajectories and possible action-response development, while retaining
their material, clock, scale and single-study limits. They are not an additional
KDP cohort, reviewed KDP masks or evidence that a KDP action prevents defects.

## Additional public leads checked — 2026-11-13

- [Lee et al., 2016 KDP droplet nucleation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5137746/)
  links three supplementary movies. The current official S3 listing for
  `PMC5137746.` returned zero keys; version-1 metadata returned 404. No movie
  payload or applicable training-reuse terms were verified. This remains a lead,
  not acquired training data; it does not establish universal inaccessibility.
- [Jiang et al., 2022 Mask R-CNN study](https://ieeexplore.ieee.org/document/9902343/)
  uses beta-L-glutamic acid, not KDP. No new KDP image/label payload was verified.
- [OpenCrystalData's primary description](https://www.sciencedirect.com/science/article/pii/S2772508124000127)
  identifies cephalexin impurity, agrochemical, polystyrene and wollastonite/LGA
  tasks. Those are not a newly verified KDP mask dataset; no payload was added.

Response bytes/status/hashes remain in
`data/audits/kdp_public_discovery_20261113_v1` and
`data/audits/kdp_public_discovery_20261113_v2`. The legacy PMC OA service returned
404; [official documentation](https://pmc.ncbi.nlm.nih.gov/tools/oa-service/)
confirms its 2026 retirement. Current bucket listing, rather than that old
endpoint alone, establishes the recorded empty distribution result. No login,
purchase, author contact or access-control bypass was used.

While KDP outline review remains pending, the separate [published-mask pilot](PUBLISHED_MASK_PILOT.md)
now trains with existing author labels. Its protein task and poor crystal result
are retained; it supplies no KDP labels or model promotion.
