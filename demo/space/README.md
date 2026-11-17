---
title: Second-life garment routing
emoji: 👕
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# Second-life garment routing

Upload a **front** photograph of a used garment (back optional). The front-only ConvNeXt-Tiny model estimates condition and damage; a rule router turns those signals into a route. When calibrated usage confidence is below the validation-chosen threshold, the final route is **HUMAN_REVIEW**.

The UI shows the **final route**, then the **model usage class** beside the **rule router route**, plus the expert sorter’s usage and condition on the held-out examples.

Weights: [`MadnessOverloaded/garment-routing-front`](https://huggingface.co/MadnessOverloaded/garment-routing-front).  
Code: [`MadMods96/garment-routing`](https://github.com/MadMods96/garment-routing).

Photographs and labels: Clothing Dataset for Second-Hand Fashion, Zenodo DOI [10.5281/zenodo.13788681](https://doi.org/10.5281/zenodo.13788681), **CC-BY 4.0** (Wargön Innovation, Myrorna, RISE / CISUTAC). Example garments are held-out test-set images from that archive.
