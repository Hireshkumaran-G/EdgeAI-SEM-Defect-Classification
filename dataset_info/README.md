# Dataset Info

## Phase 1 Collected Dataset
The full SEM dataset is not included in this repository due to size constraints
and hackathon submission guidelines.

Dataset structure used in this project:
```text
dataset/
├── train/      (each defect class 150 images)
├── val/        (each defect class 32 images)
└── test/       (each defect class 33 images)
```
Each split contains class-wise subfolders for all defect categories:
- clean
- bridge
- cmp
- crack
- ler
- open
- other
- particle
- scratch
- vias

The complete dataset is provided separately through the hackathon submission portal.

---

## Phase 2 Evaluation Dataset (Hackathon Provided)

For Phase 2, a separate `hackathon_test_dataset` was provided by the organizers.

Important notes:
- The Phase 2 dataset may not fully match the Phase 1 defect classes.
- No retraining was performed (as per rules).
- Mismatched classes were naturally classified under "other".
- The full Phase 2 dataset is not included in this repository.
