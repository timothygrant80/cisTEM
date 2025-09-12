# 2DTM Postprocessing

A modular Python package for postprocessing 2D template matching results from cryo-EM workflows (e.g., cisTEM), including 2DTM p-value calculation, particle extraction and filtering.

---

## Installation

```bash
git clone https://github.com/kekexinz/2DTM_postprocess_tool.git
cd 2DTM_postprocess_tool
pip install -e . # editable mode
```

## 📦 Usage

### `extract-particles`
Extract initial particle peaks from 2DTM search.
```bash
extract-particles \ 
--db_file <cistem.db> \
--tm_job_id 1 \
--ctf_job_id 1 \
--pixel_size 1.0 \
--output <extracted_peaks.star>
[--metric pval] \ # "zscore" or "pval"
[--metric_cutoff 8.0] \
[--threads 22] \
[--local_max_filter] \ # "snr" or "zscore" (default) used for skimage peak_local_max
[--min_peak_radius 10] \ # used for "min_distance" in skimage peak_local_max
[--exclude_borders 92] \ # avoid finding partial particles near the edge of the image, used for skimage peak_local_max
[--quadrants 1] \ # 1 (default) or 3, calculating p-value for only the first-quadrant or quadrant 1,2,4 (recommended for small particles) 

```

### `filter-particles`

Filter particles based on image thickness and/or angular invariance.

```bash
filter-particles \
  --star_file <extracted_peaks.star> \ # output from extract-particles
  --db_file <cistem.db> \
  --tm_job_id 1 \
  --ctf_job_id 1 \
  --pixel_size 1.0 \
  --output filtered_peaks.star \
  [--avg_cutoff_lb] \ # angular search CC per-pixel avg
  [--sd_cutoff_ub] \ # angular search CC per-pixel sd
  [--snr_cutoff_ub] \ 
  [--filter_by_image_thickness] \ # ctffind5 parameters
  [--thickness_cutoff_lb] \
  [--thickness_cutoff_ub] \
  [--ctf_fitting_score_lb] \
  [--ctf_fitting_score_ub] \
```

### 3D reconstruction & refinement in cisTEM
The output extracted_peaks.star and filtered_peaks.star can be imported into cisTEM as a RefinementPackage for further 3D reconstruction and refinement.
