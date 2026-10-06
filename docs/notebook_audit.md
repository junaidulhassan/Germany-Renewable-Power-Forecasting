# Notebook audit

What the original notebook (`notebooks/manisha-project_germany.ipynb`) does, which numbers it
reports, and where the pipeline in this repository deliberately behaves differently. Every number
below comes from outputs stored in the notebook, or from `scripts/rescore_saved_predictions.py`.
Nothing was re-run on a GPU.

## What the notebook does

1. **Targets.** Reads the ENTSO-E generation CSV (15-minute), keeps `solar`, `wind_onshore`,
   `wind_offshore` for 2023-2024 and imputes the 12 + 12 missing values with `IterativeImputer`.
2. **Features.** Downloads DWD HOSTRADA monthly NetCDF files, keeps every second grid point
   (938 x 720 to 469 x 360) and pickles the hourly arrays.
3. **Models.** One CNN per target on a single weather frame (radiation for solar, wind speed for
   wind): `Conv(32) - Pool - Conv(64) - Pool - Flatten - Dense(128) - Dropout(0.5) - Dense(1)`,
   82,922,113 parameters, Adam, MSE, early stopping, trained on 2023.
4. **Reporting.** R² on a random 10% hold-out of 2023, then plots of the first 250 / 300 hours of
   the 2024 forecast against the observations.

## Reported numbers (from the notebook outputs)

| Target | Training hours | Validation hours | Epochs run | Val. MAE | Val. R² |
|---|---|---|---|---|---|
| Solar | 7,884 | 876 | 11 of 15 (early stop, best = 8) | 801 MW | 0.9767 |
| Wind | 6,566 | 730 | 50 of 50 | 1,686 MW | 0.9559 |

Machine-readable copy: [`notebook_results.json`](notebook_results.json).

## Findings

### 1. The headline R² is a shuffled-hour hold-out, not a forecast score

`train_test_split(..., test_size=0.1, random_state=42)` shuffles hours. Weather and generation change
slowly from one hour to the next, so almost every validation hour has training hours within one or
two hours of it on both sides. The reported 0.9767 / 0.9559 therefore measure interpolation between
neighbouring hours of 2023, not prediction of a new period. The notebook never computed a score for
its 2024 forecasts.

Re-scoring the saved 2024 predictions (`solar_power.csv`, `wind_power.csv`) against the observations
gives the out-of-sample picture:

| Target | Alignment | Hours | R² | MAE (MW) | RMSE (MW) | nMAE |
|---|---|---|---|---|---|---|
| solar | notebook (local-time labels) | 7294 | 0.854 | 2,718 | 4,464 | 0.326 |
| solar | UTC-aligned | 7295 | 0.931 | 1,855 | 3,074 | 0.222 |
| wind | notebook (local-time labels) | 7295 | 0.841 | 3,257 | 4,233 | 0.218 |
| wind | UTC-aligned | 7295 | 0.845 | 3,160 | 4,172 | 0.212 |

(nMAE = MAE divided by mean observed generation. Table generated into
[`saved_predictions_rescore.md`](saved_predictions_rescore.md).)

### 2. Generation is in local time, the weather grids are UTC

The generation CSV repeats the hour `02:00-02:59` on 2023-10-29 and 2024-10-27 (4 duplicated
quarter-hours each, 8 in total) and skips an hour in March: it is German local time with daylight
saving. HOSTRADA time stamps are UTC. The notebook joined them as naive strings
(`df['MTU'].isin(time_df['MTU'])`), so from late March to late October a weather frame is paired with
generation from one or two hours *earlier* in the day.

The last two rows of the table above are the evidence: the same saved predictions score
R² 0.854 against mislabelled truth and **0.931** against truth converted to UTC. Wind, which varies
slowly through the day, barely moves (0.841 to 0.845). Because the models were also *trained* on
misaligned pairs from late March to late October 2023, retraining with aligned data should help further; that
is not yet measured.

### 3. The 2024 inputs were scaled with statistics from 2024

Before predicting, the notebook fitted a new `MinMaxScaler` on the 2024 weather data instead of
reusing the training scaler. The same pixel therefore maps to different network inputs in training
and in inference. The pipeline fits the scaler on training hours only and stores it with the run.

### 4. The wind model never saw November-December

The wind training array has 7,296 hours (2023-01-01 to 2023-10-31) while solar has 8,760. The
network is evaluated on 2024, which includes winter wind. Check that the 2023 wind files for
November and December are available before retraining; the pipeline reports how many hours it
aligned.

### 5. Smaller points

* The saved solar predictions contain a constant ~24.4 MW at night (observed generation is near 0).
  The linear output layer has no floor; the pipeline clips predictions at zero and standardises the
  target so the output bias is learned on a sensible scale.
* Targets are raw MW, so the loss is of order 10^7 and the Keras `mae` metric is in MW. Standardising
  the target makes learning rates and early stopping behave predictably.
* The wind direction field is downloaded and mapped but not used by any model. A circular variable
  needs sin/cos encoding before it can be a CNN channel.
* Validation loss is lower than training loss in both training-history plots. That is expected with
  `Dropout(0.5)` active only during training, and with a shuffled validation set drawn from the same
  hours as the training set.
* The solar validation loss jumps up at epochs 5 and 9 (see the training-history plot), so the
  curve is noisy rather than steadily improving; an unscaled loss of order 10^7 is a plausible
  contributor, though this was not isolated.
* All arrays are held in RAM (the notebook states 32 GB minimum, 64 GB used). The pipeline streams
  from disk instead.
* Weather covers Germany only. If the generation series covers a larger bidding zone than the weather
  grid, part of the target cannot be explained by the inputs. Worth confirming for the CSV in use.
* Installed capacity grows year over year; a model trained on 2023 and tested on 2024 will tend to
  under-predict peaks unless capacity is accounted for. The 2024 forecast plots do show under-predicted
  peaks, but this was not isolated as the cause.
