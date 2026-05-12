# MLP Attribution Sanity

This note summarizes the current-state MLP attribution check on the SWaT
validation-overlap export. The goal is a narrow sanity check before symbolic
regression, not a claim that the learned model has recovered a full physical
law.

## Input Space

MLP attribution must use normalized inputs because the model was trained on
normalized inputs. The gradient path therefore uses the MLP export tensor
shaped `[N, 51, 1]`, aligned to the distillation overlap by taking the last
`N_overlap` samples.

Do not feed `distill_inputs_current_raw.npy` directly into the trained MLP for
backpropagation. Those raw values are outside the coordinate system used during
training.

## Attribution Views

Normalized gradients are useful for model-internal dependency diagnostics. The
saved normalized-gradient matrix contains:

```text
mean_n(abs(d y_norm_j[n] / d x_norm_i[n]))
```

This is computed per sample before averaging. It is not the absolute gradient
of a batch-summed output, because opposite signs across samples can cancel.

Raw-unit sensitivities are also saved. They apply the chain rule:

```text
d y_raw_j / d x_raw_i =
  d y_norm_j / d x_norm_i * safe_std[target_feature_idx_j] / safe_std[input_feature_idx_i]
```

These raw sensitivities are mathematically valid under the saved
normalization. However, they can be misleading as top-k feature rankings when
input channels use artificial standard-deviation floors.

## Floored SWaT Channels

For this SWaT MLP checkpoint, several static or near-static pump channels have
saved `std == std_floor == 0.01`:

```text
P102, P201, P202, P204, P206, P401, P403, P404, P502, P601, P603
```

Because raw sensitivity multiplies by `std_target / std_input`, these floored
channels can receive very large raw-unit sensitivity values. The raw
sensitivity matrices are therefore diagnostic outputs, not the recommended
source for top-k feature selection when floored channels are present.

Nonfloored ranking files mask these channels while preserving the original
full ranking files unchanged.

## Delta Correlation Fallback

Delta correlation is a useful fallback for physical sanity checks. It measures
absolute Pearson correlation between raw current inputs and MLP-predicted raw
deltas. Constant feature or target columns are assigned correlation zero, so
the ranking does not produce NaNs for static channels.

For LIT101, the MLP-predicted delta correlation diagnostic is partially
physically plausible:

```text
MV101, FIT101, FIT301, P302, DPIT301, LIT101, MV302, AIT203, P101, AIT503
```

`FIT101` is high-ranked. `FIT201` appears but is weaker, just outside the top
10 in this run. This supports only a limited LIT101 sanity check; it is not
enough evidence to claim broad GeCo-style agreement.

## Sensor Mapping

SWaT sensor targets are not simply `features[:25]`. Target-to-feature mapping
must always go through `sensor_idx`:

```text
target_columns[j] == feature_columns[sensor_idx[j]]
```

This matters for raw sensitivity conversion, delta identity subtraction, and
all target-specific rankings.

## Current Recommendation

For first LIT101-only PySR experiments, use three input sets:

1. manually specified physical variables,
2. all current-state variables,
3. attribution-guided variables from MLP-predicted delta correlation with
   floored channels excluded.

Use normalized gradients as a model-internal diagnostic. Keep raw-unit
sensitivity outputs for inspection, but do not use them as the default top-k
selector while floored channels are present.

GRU attribution is deferred because it requires attribution over `[N, 51, 60]`
windows and a separate decision on how to aggregate time, feature, and
temporal-summary contributions.
