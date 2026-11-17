---
license: mit
library_name: pytorch
tags:
- image-classification
- fashion
- circular-economy
- garment-routing
pipeline_tag: image-classification
---

# Second-life garment routing (front-only)

Used textiles are sorted by hand into a next life: reuse, export, repair, remake, or recycling. This ConvNeXt-Tiny model looks at a **front** photograph, estimates condition and damage, and feeds a rule router that can abstain to human review. Built as a TU/e EngD portfolio prototype for digital textile ecosystem identification (Madhusudan Pareek). It is research / decision-support, not a certified sorter.

Pack: front-only fusion, image size 384, float16 weights, validation-fitted temperature 2.045, abstain threshold 0.494.

Code: [MadMods96/garment-routing](https://github.com/MadMods96/garment-routing).

## Data

Clothing Dataset for Second-Hand Fashion v3, Zenodo [10.5281/zenodo.13788681](https://doi.org/10.5281/zenodo.13788681), **CC-BY 4.0**. Expert labels from Wargön Innovation (stations 1–2) and Myrorna (station 3). Training uses stations 1–2 only. Station 3 and the gold set never enter training or temperature / threshold fitting.

## Best vs other (real eval)

Chosen model: **front-only** (test usage macro-F1, 4 training classes).

| Model | Test usage macro-F1 (4-class) | Station 3 usage macro-F1 |
| --- | --- | --- |
| **front-only (best / this repo)** | 0.343 (95% CI 0.294–0.403) | 0.188 |
| two-view concat | 0.300 | 0.223 |
| majority baseline | 0.164 | 0.247 |

Attribute LightGBM ceiling on test (photos unused, 6 expert classes): 0.405.

## Front-only detail

| Split | Usage macro-F1 (4-class) | Balanced accuracy | Condition MAE | Condition within ±1 | ECE after T |
| --- | --- | --- | --- | --- | --- |
| test | 0.343 (95% CI 0.294–0.403) | 0.378 | 0.995 | 0.741 | 0.047 |
| station 3 | 0.188 | 0.356 | 1.010 | 0.753 | 0.075 |

### Gold-set Cohen's kappa (garment-level 95% bootstrap CIs)

- Expert versus expert, usage: 0.193 (95% CI 0.119–0.271)
- Expert versus expert, condition (quadratic): 0.308 (95% CI 0.202–0.410)
- Model versus Wargön (Wargön photo), usage: 0.079 (95% CI 0.017–0.152)
- Model versus Myrorna (same Wargön photo), usage: 0.113 (95% CI 0.056–0.185)

Myrorna's gold photographs are missing from the archive, so its annotations are scored using the Wargön photograph of the same garment.

## Intended use

Research / portfolio demo for second-hand sorting decision support. A person remains responsible for the garment. Not for unattended sorting, EPR evidence, fibre ID from photos, or brand authentication.

## Limitations

- Router scores in `configs/router.yaml` are illustrative placeholders, not LCA or partner capacity.
- Repair / Remake / Energy recovery collapse to the training class Other.
- Expert disagreement is large; read model kappa next to expert-versus-expert kappa.
- Station 3 checks a second Swedish site, not another country.
- Absolute macro-F1 is modest; the claim is careful evaluation under label noise, not SOTA accuracy.
- CPU demo loads float16 weights and runs float32 inference.

## Files

- `model.fp16.pt` — state dict (float16), config, temperature, abstain threshold
- `router.yaml` — illustrative routing weights

## Citation

Dataset: Farrukh Nauman, RISE, Wargön Innovation AB, Myrorna AB, Vinnova / EU CISUTAC. CC-BY 4.0.

