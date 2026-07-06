# temperature_sensitivity

## Draft Caption

Development-set temperature sensitivity for post-hoc temporal softmax calibration. Curves show NRMSE on the development split as logits are converted to event-time posteriors with different softmax temperatures. Vertical guide lines mark the selected temperature for each run. The plot is a diagnostic for how strongly each learned logit field must be sharpened or smoothed before scalar readout.

## Selected Temperatures

- CE: tau=0.70, development NRMSE=0.9223.
- CE+time: tau=0.80, development NRMSE=0.9245.
- EventNLL: tau=0.70, development NRMSE=0.9236.
- Time-only: tau=0.85, development NRMSE=0.9353.
- Wasserstein: tau=2.95, development NRMSE=0.9369.

Diagnostic note: No selected temperature is on the grid boundary. Boundary selections should be interpreted as a warning that the calibration grid may be too narrow or that the corresponding loss produced logits whose posterior geometry requires strong post-hoc correction.
