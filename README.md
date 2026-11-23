# Cloud Formation Analyzer

Cloud Formation Analyzer is a Streamlit/OpenCV application for classical segmentation of candidate cloud pixels in sky photographs. It estimates image-pixel coverage; it is not a forecast system or an official meteorological measurement.

## Technical audit

- The input path loads PIL images as RGB; OpenCV transforms use `COLOR_RGB2HSV` / `COLOR_RGB2LAB`. No current RGB/BGR mismatch was found.
- ROI validation clamps coordinates to image bounds, then crops before feature extraction. Polygon validity is applied inside that crop.
- The previous principal failure was the fixed HSV range: color/brightness limits do not model changing illumination, so bright warm sky can resemble cloud and dark/thin cloud can be omitted.
- Gaussian blur and opening/closing can suppress narrow cloud boundaries, depending on kernel size. The adaptive workflow defaults to no morphology and exposes refinement controls.
- Coverage now divides final cloud pixels by the count of valid ROI/polygon pixels.
- Ground-truth masks must exactly match the ROI shape. Evaluation excludes invalid polygon pixels; it does not resize or align annotations.

## Install and run
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Workflow
1. **Overview:** project and process summary.
2. **Image Input:** upload a JPG/JPEG/PNG, set the rectangular ROI, and optionally enter polygon points.
3. **Processing:** set algorithm thresholds and weights, optionally mark a sunlight-interference rectangle in ROI-local coordinates, then run all methods on the same ROI.
4. **Segmentation:** compare the original, existing structure-aware baseline, thin-cloud candidates, dark-cloud candidates, adaptive fusion, and ambiguous regions.
5. **Analysis:** use Analysis Mode immediately for selected-method coverage, visual comparisons, feature observations, and ambiguity; switch to Evaluation Mode to score every compatible method against a matching ground-truth mask.

## Methods

### Baselines: HSV, LAB, Otsu
- **HSV** is the original configurable color threshold baseline. OpenCV hue is 0–180; saturation and value are 0–255.
- **LAB** selects configurable lightness and chroma ranges.
- **Otsu** selects a global grayscale brightness threshold. It is included as a baseline, not described as HYTA.

### ARGD
ARGD uses normalized RGB channels and floating-point arithmetic:

`ARGD = k * R - G`

The base `k` and alternate sunlight-region `k` can be adjusted. A marked ROI-local rectangle is user input; CFA does not infer a sun location or solar angle. Image-derived brightness/warmth cues are diagnostics, not solar-position estimates.

### NSV
NSV is computed after RGB-to-HSV conversion with normalized saturation/value channels:

`NSV = (S - V) / (S + V + epsilon)`

`S` and `V` are scaled to [0,1]. Epsilon makes black pixels finite. Lower feature values are cloud candidates in this implementation. The threshold is user-adjustable. See Jayadevan et al., DOI [10.1175/JTECH-D-14-00053.1](https://doi.org/10.1175/JTECH-D-14-00053.1).

### HYTA-inspired normalized blue/red ratio
The feature is computed from normalized RGB:

`NBR = (B - R) / (B + R + epsilon)`

The histogram is returned with its standard deviation. Modality selection is an explicit, configurable standard-deviation boundary: lower dispersion uses a fixed threshold; higher dispersion uses iterative Li-style minimum-cross-entropy thresholding. This is a transparent HYTA-inspired implementation, not a reproduction of the paper's exact parameterization or benchmark. See Li et al., DOI [10.1175/JTECH-D-11-00009.1](https://doi.org/10.1175/JTECH-D-11-00009.1).

### Superpixel-local threshold and structure-aware segmentation
The project installs `opencv-contrib-python-headless` so OpenCV `ximgproc` SLIC is available. SLIC count and compactness are adjustable. Watershed is an explicit alternative and its diagnostics are labeled accordingly; it is never reported as SLIC.

The **Texture + SLIC** and **Structure-aware** methods extract RGB, HSV, ARGD, NSV, normalized blue/red ratio, local standard deviation, local entropy, and gradient magnitude. NSV retains the research feature definition used by this project:

`NSV = (S - V) / (S + V + epsilon)`

Here, S and V are normalized to [0,1], with a small positive epsilon for numerical stability. The three texture maps are robustly scaled using valid-ROI percentiles, then combined with threshold-relative color evidence using configurable weights. Each SLIC region is classified from its mean evidence, not from independent pixel decisions. The **Structure-aware** method additionally gives a bounded bonus to near-threshold neighboring regions only when their texture evidence supports continuity; this is intended to retain shadowed portions without treating every connected region as cloud.

The optional warm-bright / user-marked sunlight penalty is applied at region level. It is an experiment control, not a solar detector. The configurable classification threshold is an evidence-score cutoff; scores and ambiguous masks are heuristic and are not calibrated probabilities. Changing texture window size affects the scale of detected variation; increasing the classification threshold or reducing continuity favors fewer selected regions.

The original NSV-local threshold method remains available as **Superpixel**. It uses each region's NSV median/MAD and a conservative candidate ceiling.

### Thin- and dark-cloud enhancement
The enhancement stage reuses the pipeline's SLIC labels and existing color/texture maps. Local contrast adds a signed gray-level residual, a contrast normalized by local standard deviation, and a local adaptive threshold (`local mean + scale × local standard deviation`). Thin-cloud candidates must combine positive local contrast with cloud-like NSV/blue-red evidence and regional texture or baseline support; ordinary contrast variation alone is not selected. Candidate decisions remain pixel-level inside supported regions so thin filaments are not reduced to one whole-superpixel label.

Dark-cloud candidates combine low-saturation/neutral color evidence with local standard deviation, entropy, gradient/edge evidence, and negative local contrast. A region must have texture/edge evidence or be adjacent to an existing baseline cloud region. Thus a uniform dark blue sky region is not selected merely because it is dark, and a neutral dark region without structural or neighboring evidence is rejected.

**Adaptive fusion** preserves the existing structure-aware baseline mask. Enhancement candidates are blended by their configured weights, then added only when their heuristic score clears the fusion threshold. The thin and dark masks are not blindly ORed into the baseline. Enhancement outputs bypass morphological opening so filament candidates are not erased by post-processing; the baseline continues to use the configured refinement operation.

These are classical, image-only heuristics. Scores and ambiguous maps are not calibrated probabilities. Texture-free cloud interiors, transparent cirrus that has no measurable local contrast, and a uniformly dark overcast image with no visible boundary may remain indistinguishable from clear sky.

### Legacy hybrid score and ambiguity
The hybrid combines threshold-relative ARGD, NSV, normalized blue/red, and local-region evidence into a weighted score. It does not OR the masks together. Configurable weights are normalized by their sum. Texture can provide limited supporting evidence. Warm-bright and user-marked sunlight regions can reduce the score by a configurable penalty. Pixels near the decision boundary are shown in an ambiguity map; this does not assert that they are cloud or non-cloud with certainty.

The hybrid weighting, warm-sky cue, dispersion boundary, and thresholds are experimental controls, not validated universal parameters. Illumination changes and RGB-only ambiguity remain limitations.

### Texture
Optional evidence maps include local grayscale standard deviation, local entropy, and gradient magnitude. Texture is supporting evidence only; smooth cloud and thin cirrus can have low texture.

## ROI and coverage
The rectangular ROI preserves source aspect ratio and excludes pixels outside the crop. A full-image-coordinate polygon can further exclude non-sky areas inside it. A rectangle alone does not remove every building, tree, horizon, or glare region; inspect the preview and mask.

`Cloud coverage (%) = final cloud pixels / valid sky pixels * 100`

Pixels outside the valid-sky mask are excluded. The reported coverage is an estimate based on this image and selected thresholds.

## Evaluation workflow
**Analysis Mode** works without a ground truth. It reports cloud/sky coverage and counts only inside valid ROI pixels, lets you select a result, compares all compatible methods, and summarizes extracted brightness, RGB/HSV, texture, gradient, spatial distribution, connected regions, candidate masks, and ambiguity. Coverage is not an accuracy measure. The no-ground-truth recommendation intentionally abstains because heuristic internal scores cannot establish which method is more accurate.

**Evaluation Mode** requires a manually annotated mask for the same image and exact ROI. Confirm image correspondence and alignment before scoring. White (255) means cloud; black (0) means non-cloud. Dimensions must exactly match; CFA does not resize or prove alignment. It computes IoU, precision, recall, F1, accuracy, and TP/FP/FN/TN for every available compatible method, excluding invalid polygon pixels. All methods use the same annotation and valid-pixel mask. Methods can be ranked by a selected metric, with F1 as the default. This is a result for the current image only, not a dataset-wide winner. Separate thin/dark scores require separate region-specific annotations; without them, those results are visual observations only.

Choose the condition (clear blue, midday cloud, warm clear, golden-hour cloud, sunset cloud, haze, cirrus, or strong sunlight) and add results to the session evaluation log. The log can be downloaded as CSV. Use the same image/ROI and reference for method comparisons. The log is held in Streamlit session state; export it before ending the session to retain it.

No sample benchmark dataset or annotations are included. Therefore no real-world method comparison or accuracy improvement is claimed. Synthetic tests validate numerical behavior and accounting only.

## Research references
- Li et al. (2019), sunlight-interference reduction, DOI [10.3390/atmos10110640](https://doi.org/10.3390/atmos10110640).
- Jayadevan et al. (2015), NSV feature, DOI [10.1175/JTECH-D-14-00053.1](https://doi.org/10.1175/JTECH-D-14-00053.1).
- Li et al. (2011), HYTA, DOI [10.1175/JTECH-D-11-00009.1](https://doi.org/10.1175/JTECH-D-11-00009.1).
- Niu et al. (2024), clear-sky library and superpixel local threshold, DOI [10.1016/j.renene.2024.120452](https://doi.org/10.1016/j.renene.2024.120452).
- Hasenbalg et al. (2020), algorithm benchmarking, DOI [10.1016/j.solener.2020.02.042](https://doi.org/10.1016/j.solener.2020.02.042).

These are methodological references only. CFA does not reproduce their datasets, exact experimental setup, or reported performance.

## Tests
Run the synthetic suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
