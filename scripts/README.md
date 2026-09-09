# Scripts (Reserved)

This directory will contain the operational entry points implied by the
project roadmap. Currently empty — listed here so names are agreed on
before implementation begins.

## Planned scripts

| Script | Purpose | Phase |
|---|---|---|
| `record_teleop.py` | Launch teleop GUI, drive the robot, save raw episode | Phase 2 |
| `convert_raw_to_processed.py` | Clean raw recordings → LeRobotDataset format | Phase 3 |
| `augment_resim.py` | Re-simulate teleop commands against randomized conditions | Phase 4 |
| `augment_mirror.py` | Left-right mirror existing episodes | Phase 4 |
| `augment_timewarp.py` | Resample episodes at different speeds | Phase 4 |
| `generate_scripted_demos.py` | Procedural demonstration generation | Phase 6 |
| `train_bc_local.py` | Small BC model training (ACT) on CPU/MPS | Phase 5 |
| `push_dataset.py` | Publish dataset version to HF Hub | Phase 3+ |
| `push_model.py` | Publish model checkpoint to HF Hub | Phase 5+ |
| `evaluate_policy.py` | Roll out a policy in sim, measure success rate | Phase 5+ |
| `update_manifests.py` | Regenerate `data/manifests/` indexes | Phase 3+ |
