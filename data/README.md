# Data

Nothing in this directory is committed except this file and the `.gitkeep` placeholders. All data is
large and can be recreated.

## Layout

```
data/
├── raw/
│   ├── generation/   20200101-20241030-gen_per_prod_type.csv   <- you provide this
│   └── weather/      <variable>/*.nc                           <- `renewcast download`
├── interim/          targets_hourly.csv                        <- `renewcast prepare-targets`
└── processed/
    └── weather/      <variable>.npy, <variable>_time.npy, ...  <- `renewcast preprocess`
```

## Sources

### Generation (targets)

`20200101-20241030-gen_per_prod_type.csv`: actual generation per production type for the German
bidding zone (DE-AT-LU), 15-minute resolution, 2020-01-01 to 2024-10-30, published by the
transmission system operators through the ENTSO-E Transparency Platform. The original project used a
Kaggle copy of this file. Columns used: `MTU`, `solar`, `wind_onshore`, `wind_offshore` (MW).

* `MTU` is **local time** (CET/CEST). Every October the same hour appears twice; the pipeline
  converts to UTC using the order of the duplicates.
* 12 `solar` and 12 `wind_onshore` values are missing in 2023-2024 and are imputed.

Place the file at `data/raw/generation/20200101-20241030-gen_per_prod_type.csv`, or change
`paths.raw_generation` in the config.

### Weather (features)

DWD Climate Data Center, HOSTRADA hourly gridded data for Germany (about 1 km grid, 938 x 720 cells):
<https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/>

| NetCDF variable     | DWD directory           | Meaning                            | Used for         |
|---------------------|-------------------------|------------------------------------|------------------|
| `rsds`              | `radiation_downwelling` | Downwelling shortwave radiation    | solar model      |
| `sfcWind`           | `wind_speed`            | Near-surface wind speed            | wind model       |
| `sfcWind_direction` | `wind_direction`        | Near-surface wind direction        | EDA map only     |

One file per month and variable, for example
`sfcWind_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023013123.nc`. Time stamps are UTC. Only the
`wind_direction` directory name appears in the original notebook; the other two follow the DWD
naming scheme and should be confirmed with `renewcast download --dry-run`.

Expect roughly 10 GB for 2023-2024 across the variables, and about 6 GB of `.npy` per variable per
year at the default downsampling (469 x 360 float32 per hour).

## Files from the original notebook

These were produced by the notebook and are *not* inputs to the pipeline:

| File                  | Content                                                              |
|-----------------------|----------------------------------------------------------------------|
| `dataset_2023_24.csv` | cleaned 2023-2024 targets, local-time labels, 15-minute              |
| `solar_power.csv`     | the notebook's saved 2024 solar predictions (hourly)                 |
| `wind_power.csv`      | the notebook's saved 2024 wind predictions (hourly)                  |

`scripts/rescore_saved_predictions.py` scores the two prediction files against the observations.
