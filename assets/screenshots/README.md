# Screenshot library

Screenshots of the real application on the tutorial cohort (Start > Load the tutorial cohort), with the
clinical-anomaly and study-inclusion criteria ticked (928 of 1,206 rows). Each one is a 1440x900 browser viewport
captured at 2x and stored as lossless WebP. They are the source of the images of the documentation site
(`docs/assets`) and of the README (`assets/readme`), and of any new image, such as another animation.

| Screenshots | What they show |
| :-- | :-- |
| `start`, `workflow_start` | The Start page just after loading, and its workflow map |
| `quality`, `population` | Data > Validate: the quality score, the analysis population and its participant flow |
| `explore_types`, `explore_stats`, `explore_qual` | Explore > Statistics & correlations: column types, quantitative and qualitative statistics |
| `explore_graph`, `explore_matrix` | The correlations as a network and as a matrix |
| `visualize` | Explore > Visualize: a box plot with p-values between groups |
| `plot_*` | Charts on their own: `box`, `violin`, `scatter`, `corr`, `clustermap`, `sunburst`, `network`, `explore_matrix` |
| `mi_setup`, `mi_diagnostics` | Data > Enrich: a multiple imputation (m = 10) and its diagnostics |
| `survival`, `cox` | Analyze > Survival & regression, pooled across the 10 imputed datasets |
| `plan`, `comparison` | Analyze > Hypothesis tests & epidemiology: the plan of a group comparison, then its results |
| `dossier`, `dossier_results` | Reproduce > Study dossier: the participant flow and the results added to it |
| `traces` | Reproduce > Versions & traces |
| `start_workflow`, `workflow` | The Start page at the end, and its workflow map with the steps done |

To capture them again:

1. Start the application with a throwaway data folder and no sign-in:
   `CMP_IAM_PROVIDER=disabled DATA_ROOT=<tmp> uv run streamlit run main.py --server.port 8769`
2. Capture: `CMP_SITE_BASE=http://localhost:8769 uv run --no-project --python 3.14 --with playwright python dev_scripts/site_screenshots.py <out-dir>`
3. Keep them here: `uv run python dev_scripts/keep_screenshots.py <out-dir>`
4. Build the images: `uv run python dev_scripts/make_readme_assets.py` and `uv run python dev_scripts/make_site_assets.py`
