# Data

## Primary cohort — MetaboNet (T1D, n = 1,092)
- **Download** MetaboNet v2.0 from <https://metabo-net.org> (dataset paper: [15] in manuscript; *J Diabetes Sci Technol*, 2026).
- **Place** the processed file at `data/metabonet_v2/metabonet_v2_data.npz` (1,092 subjects; 60/20/20 subject-level split).
- Preprocess raw traces with:
  ```bash
  python src/process_metabonet_v2.py
  ```

## Cross-cohort validation — Shanghai T2DM (T2D, n = 100)
- **Source**: ShanghaiT1DM/ShanghaiT2DM datasets, *Scientific Data* 2023 (doi:10.1038/s41597-023-01940-7) [25], downloadable from figshare.
- **Place** the processed file at `data/t2d/t2d_metabonet_data.npz` (100 subjects; 63/21/19 subject-level split).
- The T2D sequence generation follows the same preprocessing parameters as T1D (lookback 12, horizon 6, 5-min sampling, stride 1, windows capped at 1,000/subject).

Files are large and not committed; place them under `data/` (ignored by `.gitignore`) before running the pipeline.
