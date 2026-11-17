# Second training run (~10k train)

## Why two runs

1. **Ablation on the current ~6k subset** (`configs/front_only_384.yaml`): image size 384 + focal holes loss + Major threshold. Any lift vs the old 224 front checkpoint is from loss/resolution, not more data.
2. **Scale-up** after the Zenodo fetch finishes: retrain the same recipe on ~10 018 train garments.

That split is the interview-friendly ablation.

## Subset math

Under the existing 70/15/15 Wargön split:

| Piece | Count |
|-------|------:|
| `subset_size` | 16 200 |
| train | ~10 018 |
| val / test | ~2 143 / ~2 140 |
| station3 cross-site | 1 500 |
| gold | 399 |

Member list for download: `data/subset_members_16200.txt` (32 400 front+back paths).  
Active train still uses `data/subset_members.txt` until resize is done.

## Fetch (started in parallel)

```bash
python scripts/fetch_subset.py --members data/subset_members_16200.txt --workers 1 --min-interval 0.4
```

Resumable (skips non-empty files). Log: `data/fetch_16200.log`. Expect ~5–6 hours for the remaining ~25k files at 0.4 s spacing.

## After fetch completes

Automated path (already started as a waiter):

```bash
python scripts/after_fetch_16200.py
```

That resizes `subset_members_16200.txt`, then runs:

```bash
python scripts/eval_expanded_dual.py
```

which scores **both** `runs/convnext_tiny_front/best.pt` (224) and `runs/convnext_tiny_front_384/best.pt` (384, once present) on the expanded val/test garments (same splits, more images on disk). It reports holes **Major** recall with bootstrap CIs and test **HUMAN_REVIEW** share.

Manual:

```bash
python scripts/resize_images.py --members data/subset_members_16200.txt
python scripts/eval_expanded_dual.py
```

Do not replace `subset_members.txt` for training until resized images for the new IDs exist under `data/resized`.
