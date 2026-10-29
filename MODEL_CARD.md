# Model card

## Intended use

Decision support for research on second-hand garment sorting. Given a front photograph, a back photograph, and an optional mock product passport, the system estimates condition, pilling, stains, and holes, then suggests a next route: reuse, export, repair, remake, recycle, energy recovery, or human review.

The audience is a researcher or a sorting team looking at a portfolio prototype. A person remains responsible for the garment.

## Out of scope

- Unattended sorting for resale, export, or waste shipment
- Extended producer responsibility evidence or a legal product passport
- Fibre identification from the photograph
- Brand authentication, pricing, or identification of people
- Facilities outside the Swedish sources this dataset was collected in, without a new labelled test set

## Data

Clothing Dataset for Second-Hand Fashion, version 3, Zenodo DOI 10.5281/zenodo.13788681, CC-BY 4.0. Expert sorters at Wargön Innovation (station 1 and station 2) and Myrorna (station 3) labelled the garments. Training and validation use station 1 and station 2 only. Station 3 is a cross-site test. Files under `test100`, and any other file whose comment contains `Test`, are held out of training. Annotator agreement uses the garments whose comments carry a `Test` item number. The README separates that garment count from the number of held-out JSON files.

Brand photographs are not used. Price is not a target. Stain labels from the older Yes/No protocol are excluded from the three-class stain head. Missing holes and smell stay missing.

## Metrics

All reported scores are written by `garmentroute.evaluate` into `reports/results.json`. This card does not repeat them, so it cannot drift away from that file. The metrics are:

- per head: accuracy, macro-F1, confusion matrix
- condition and pilling: also MAE and accuracy within one level
- in-distribution test and station 3, side by side
- gold set: Cohen's kappa, model versus expert next to expert versus expert; on garments labelled at both sites, cross-site kappa next to within-Wargön kappa, the Wargön-versus-Myrorna route matrix, and condition drift
- usage ECE before and after temperature scaling, fit on the validation split only
- coverage against accuracy when the model abstains
- router agreement against the expert usage label, both from predicted damage and from the expert's own damage labels
- attribute-only logistic regression and gradient boosting: macro-F1 over all 6 expert classes and over the 4 training classes, against the majority baseline and the rule router, plus permutation importance on the validation split
- station 3: balanced accuracy and macro-F1, with accuracy secondary, for the uncorrected predictions, for predictions reweighted by a Laplace-smoothed labelled sample held out of the score, and for predictions reweighted by an EM prior (Saerens, Latinne, and Decaestecker, 2002) that uses no station 3 labels

The abstention threshold is chosen on validation. It is not tuned on station 3 or on the gold set.

## Known failure modes

- Repair, Remake, and Energy recovery are rare. The vision head collapses them to Other, so a direct usage prediction cannot name them.
- The illustrative router weights can prefer a route a sorter would not. Read `configs/router.yaml` before treating a route as a recommendation.
- Expert disagreement is large enough that a model can look wrong and still sit inside the human range, or look close on accuracy and still sit below the human kappa. Use the kappa comparison, not accuracy alone.
- Lighting, blur, and the change of background (measuring tape, then a 10 cm grid) move the images. Station 3 is the check for a different site, not for a different country.
- A stain can be the colour of the print. Mild colour jitter is used in training for that reason, and it can also hide a faint stain.
- Legacy Yes/No stains are not Minor or Major. Treating them as one of those classes would invent a severity.
- If the passport fibre string is empty, high-value fibre rules do not fire. The model will not guess wool from a knit texture.
- Monte Carlo dropout entropy is only meaningful when evaluation was run with `--mc-passes` greater than zero. Temperature scaling is the calibration method that is always fit.

## Ethical notes

The brand images that showed hands or faces were removed by the dataset authors. This project does not ask for them back. Routes affect whether a garment is resold, exported, or destroyed. Low confidence is supposed to become human review rather than a silent decision.
