# Project Worklog — Image-Classification-for-PPE

This document records weekly progress for the Image-Classification-for-PPE project. Week 2 was left unchanged per request. Week 3 reflects recent engineering work (detection fixes, VLM simplifications, Label Studio orchestration) and Week 4 describes the repo-focused development plan.

---

## Week 1 — Project setup (brief)
- Created repository structure, `src/`, `scripts/`, `docs/`, `data/` and example splits.
- Implemented baseline SSD300-style model scaffolding in `src/models/ssd.py` and dataset utilities in `src/dataset/ppe_dataset.py`.

## Week 2 — Data & baseline training (unchanged)
- Dataset layout and sample data created under `data/`.
- Training and inference scripts: `scripts/train.py`, `scripts/inference.py` with documented CLI in `README.md`.
- Configuration consolidated in `configs/ppe_config.yaml`.

## Week 3 — Hybrid pipeline, detection fixes, captioning speedups, and annotation orchestration (UPDATED)
Summary: focused on correctness and latency improvements for the hybrid PPE detection + description pipeline and operationalizing annotation tooling for human-in-the-loop labeling.

What was changed / implemented
- Detection decoding fixes:
  - Replaced a mock/stubbed detection path with the real decoding step (proper prior-box decoding and coordinate transform) so detection outputs are now meaningful instead of identical placeholders.
  - Implemented IoU-based NMS and a small center-proximity duplicate-collapse routine to remove overlapping/redundant detections prior to captioning.

- Vision‑Language simplifications:
  - Switched from a multi-prompt/score-and-rerank pattern to a single, concise BLIP‑2 prompt with greedy decoding to reduce caption latency and repetition.
  - Removed the expensive caption-scoring loop and replaced it with a fast heuristic cleaner (`_clean_description`) that normalizes punctuation, removes repeated n-grams, and rewrites person pronouns to domain-appropriate terms (e.g., "worker").

- Annotation tooling / Label Studio:
  - Added/updated `start_label_studio.bat` to bind Label Studio to `0.0.0.0:8080` and provide guidance for launching an ngrok tunnel with host-header rewrite to avoid common Host/CSRF mismatches.
  - Added `docs/LABEL_STUDIO_SETUP.md` with a walkthrough for creating the Label Studio project, label config, and local storage setup (import `label_studio_tasks.json`).

 - Repository alignment and documentation:
    - Consolidated hybrid pipeline documentation inside this repository and updated README pointers so Week notes reference only in-repo artifacts.
    - Added `docs/HYBRID_MODEL_README.md` describing the hybrid architecture and usage examples.

Files touched (in-workspace)
- `src/models/hybrid_ppe_model.py` — real detection decoding, NMS, proximity filtering, simplified caption generation and `_clean_description` (implementation present in workspace).
- `start_label_studio.bat` — starts Label Studio on `0.0.0.0:8080` and contains ngrok-host guidance.
- `docs/HYBRID_MODEL_README.md` — detailed hybrid README (already present in `docs/`).
- `docs/LABEL_STUDIO_SETUP.md` — Label Studio step-by-step (already present in `docs/`).

Notes / validation required
- NMS thresholds and per-class tuning still need empirical sweeps (recommended grid search across conf_threshold × nms_threshold for a validation subset).
- Label Studio + ngrok should be validated with a live run and at least one remote annotator to confirm there are no CSRF/Host issues on the target network.

## Week 4 — Focused development plan (repo-centric)
Objective: concentrate Week 4 activity on turning the current workspace into a reproducible, testable, and annotation-ready development pipeline for Image-Classification-for-PPE.

Planned tasks (concrete, repo-specific)
1. Run & validate Label Studio pilot
   - Action: Ask permission to run `start_label_studio.bat` locally (or run it for you) to capture the ngrok tunnel URL and confirm remote access.
   - Success: Remote annotator can open the tunnel URL and annotate at least 25 tasks without CSRF/Host failures.

2. NMS & detection tuning
   - Action: Add a small tuning harness (script `scripts/tune_nms.py`) that loads a validation split, runs inference with configurable conf/nms thresholds, and reports mAP/precision/recall per class.
   - Success: Identify a working conf_threshold and nms_threshold pair for production inference (recommended default: conf=0.35–0.5, nms=0.4–0.5) and add to `configs/ppe_config.yaml`.

3. Annotation → training loop integration
   - Action: Create an import pipeline to convert Label Studio exports (VOC/JSON) into the repo's annotation format and splits; add `scripts/import_labelstudio.py` and document usage in `docs/LABEL_STUDIO_SETUP.md`.
   - Success: Produce a `data/annotations/` directory ready for training and a new `data/splits/train.txt` that includes the newly annotated images.

4. Small reproducible demo and tests
   - Action: Add a small end-to-end demo `scripts/demo_hybrid_ppe.py` (single image path → runs detection + hybrid description → prints JSON) and unit tests in `test_descriptions.py` (happy-path + one edge case).
   - Success: `python scripts/demo_hybrid_ppe.py --image data/images/sample_demo.jpg` runs and returns structured JSON; tests pass locally.

5. Documentation & checkpointing
   - Action: Update `README.md` quick-start with the Label Studio pilot instructions and demo commands, and save working checkpoints to `models/` with descriptive names.
   - Success: Documented reproducible steps to launch Label Studio, run a demo, and retrain from new annotations.

Stretch tasks (optional)
- Add a lightweight Streamlit UI (already present as `streamlit_app.py`) wiring to the hybrid model for quick QA.
- Add a tiny CI job (GitHub Actions) to run lint + unit tests on PRs.

Deliverables by end of Week 4
- Tuned NMS/default thresholds committed to `configs/ppe_config.yaml`.
- `scripts/tune_nms.py`, `scripts/import_labelstudio.py`, `scripts/demo_hybrid_ppe.py` added and documented.
- Label Studio pilot validated and a sample exported dataset imported into `data/annotations/`.
- WORKLOG entry (this file) updated and committed.

---

If you'd like, I can apply the planned Week 4 changes now (create the tuning/import/demo scripts and run quick local validations). Tell me which subset to prioritize: run the Label Studio pilot, implement the tuning harness, or scaffold the import/demo scripts first.

### Evaluation — 2025-10-11 20:47:43

- Model: `models/best_model_regularized.pth`
- Images evaluated: 5
- Overall mAP: 0.055
- Conf threshold: 0.05
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\eval_test\evaluation_results_20251011_204743.json`
- Class metrics CSV: `outputs\eval_test\class_metrics_20251011_204743.csv`
- Problem analysis: `outputs\eval_test\problem_analysis_20251011_204743.txt`


### Evaluation — 2025-10-11 20:49:10

- Model: `models/best_model_regularized.pth`
- Images evaluated: 5
- Overall mAP: 0.055
- Conf threshold: 0.05
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\eval_test_no_rescorer\evaluation_results_20251011_204910.json`
- Class metrics CSV: `outputs\eval_test_no_rescorer\class_metrics_20251011_204910.csv`
- Problem analysis: `outputs\eval_test_no_rescorer\problem_analysis_20251011_204910.txt`


### Evaluation — 2025-10-11 20:50:55

- Model: `models/rcnn_baseline_adamw.pth`
- Images evaluated: 25
- Overall mAP: 0.285
- Conf threshold: 0.05
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\eval_rcnn_with_rescorer\evaluation_results_20251011_205055.json`
- Class metrics CSV: `outputs\eval_rcnn_with_rescorer\class_metrics_20251011_205055.csv`
- Problem analysis: `outputs\eval_rcnn_with_rescorer\problem_analysis_20251011_205055.txt`


### Evaluation — 2025-10-11 20:53:08

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.266
- Conf threshold: 0.05
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\eval_rcnn_baseline_with_rescorer\evaluation_results_20251011_205308.json`
- Class metrics CSV: `outputs\eval_rcnn_baseline_with_rescorer\class_metrics_20251011_205308.csv`
- Problem analysis: `outputs\eval_rcnn_baseline_with_rescorer\problem_analysis_20251011_205308.txt`


### Evaluation — 2025-10-11 20:54:39

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.266
- Conf threshold: 0.05
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\eval_rcnn_baseline_no_rescorer\evaluation_results_20251011_205439.json`
- Class metrics CSV: `outputs\eval_rcnn_baseline_no_rescorer\class_metrics_20251011_205439.csv`
- Problem analysis: `outputs\eval_rcnn_baseline_no_rescorer\problem_analysis_20251011_205439.txt`


### Evaluation — 2025-10-11 21:37:22

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 5
- Overall mAP: 0.150
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251011_213722.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251011_213722.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251011_213722.txt`


### Evaluation — 2025-10-11 21:38:21

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 5
- Overall mAP: 0.150
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251011_213821.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251011_213821.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251011_213821.txt`


### Evaluation — 2025-10-11 21:39:49

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 5
- Overall mAP: 0.150
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251011_213949.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251011_213949.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251011_213949.txt`


### Evaluation — 2025-10-11 21:45:41

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 5
- Overall mAP: 0.150
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251011_214541.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251011_214541.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251011_214541.txt`


### Evaluation — 2025-10-11 22:19:12

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.180
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\rescorer_test\evaluation_results_20251011_221912.json`
- Class metrics CSV: `outputs\evaluation_results\rescorer_test\class_metrics_20251011_221912.csv`
- Problem analysis: `outputs\evaluation_results\rescorer_test\problem_analysis_20251011_221912.txt`


### Evaluation — 2025-10-11 22:20:54

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.185
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\baseline_test\evaluation_results_20251011_222054.json`
- Class metrics CSV: `outputs\evaluation_results\baseline_test\class_metrics_20251011_222054.csv`
- Problem analysis: `outputs\evaluation_results\baseline_test\problem_analysis_20251011_222054.txt`


### Evaluation — 2025-10-11 22:22:32

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.185
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\baseline_true\evaluation_results_20251011_222232.json`
- Class metrics CSV: `outputs\evaluation_results\baseline_true\class_metrics_20251011_222232.csv`
- Problem analysis: `outputs\evaluation_results\baseline_true\problem_analysis_20251011_222232.txt`


### Evaluation — 2025-10-11 22:36:26

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.185
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\gat_test\evaluation_results_20251011_223626.json`
- Class metrics CSV: `outputs\evaluation_results\gat_test\class_metrics_20251011_223626.csv`
- Problem analysis: `outputs\evaluation_results\gat_test\problem_analysis_20251011_223626.txt`


### Evaluation — 2025-10-11 22:43:24

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.183
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\gat_test2\evaluation_results_20251011_224324.json`
- Class metrics CSV: `outputs\evaluation_results\gat_test2\class_metrics_20251011_224324.csv`
- Problem analysis: `outputs\evaluation_results\gat_test2\problem_analysis_20251011_224324.txt`


### Evaluation — 2025-10-11 23:00:22

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.182
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\retrained_gat\evaluation_results_20251011_230022.json`
- Class metrics CSV: `outputs\evaluation_results\retrained_gat\class_metrics_20251011_230022.csv`
- Problem analysis: `outputs\evaluation_results\retrained_gat\problem_analysis_20251011_230022.txt`


### Evaluation — 2025-10-11 23:52:37

- Model: `models\rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.185
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\retrained_gat_alpha06\evaluation_results_20251011_235237.json`
- Class metrics CSV: `outputs\evaluation_results\retrained_gat_alpha06\class_metrics_20251011_235237.csv`
- Problem analysis: `outputs\evaluation_results\retrained_gat_alpha06\problem_analysis_20251011_235237.txt`


### Evaluation — 2025-10-25 19:31:09

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.000
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251025_193109.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251025_193109.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251025_193109.txt`


### Evaluation — 2025-10-25 19:33:02

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.048
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251025_193302.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251025_193302.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251025_193302.txt`


### Evaluation — 2025-10-25 19:46:52

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.039
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251025_194652.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251025_194652.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251025_194652.txt`


### Evaluation — 2025-10-25 19:51:06

- Model: `models/rcnn_baseline.pth`
- Images evaluated: 25
- Overall mAP: 0.028
- Conf threshold: 0.5
- Eval IoU threshold: 0.5

Saved files:
- Results JSON: `outputs\evaluation_results\evaluation_results_20251025_195106.json`
- Class metrics CSV: `outputs\evaluation_results\class_metrics_20251025_195106.csv`
- Problem analysis: `outputs\evaluation_results\problem_analysis_20251025_195106.txt`

