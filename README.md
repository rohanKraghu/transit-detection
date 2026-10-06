# Transit detection in TESS-like light curves

Finding transiting exoplanets is a needle-in-a-haystack detection problem: a few
per cent of stars host a detectable transiting planet, the signal is a 0.05–1%
dip lasting a few hours, and it sits on top of stellar variability ten to a
hundred times deeper. This repository is an end-to-end pipeline for that
problem — generate the photometry, detrend it, search it, classify it, and
evaluate it the way an imbalanced detection problem has to be evaluated.

**One command reproduces everything in this README:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py            # ~2 min on 4 cores
pytest                            # ~1 min, 86 tests
```

It writes `results/metrics.json`, `results/report.txt` and four PNGs to
`figures/`. Fixed seed (42), pinned dependencies, deterministic output.

---

## Headline result

2400 light curves, 4.0% of them hosting a transiting planet, 35% held out.

| Scorer | Average precision (68% CI) | ROC-AUC | Precision@20 |
|---|---|---|---|
| Random ranking (chance) | 0.040 | 0.500 | — |
| Baseline: rank by BLS **depth** | 0.097 [0.080, 0.128] | 0.751 | 0.00 |
| Baseline: rank by BLS **depth SNR** | 0.180 [0.149, 0.231] | 0.830 | 0.10 |
| **Gradient boosting on vetting features** | **0.730** [0.659, 0.799] | 0.933 | **0.95** |

**18× better than chance, 4.1× better than the strongest single-statistic
baseline.** Of the 20 candidates the model ranks highest, 19 are real planets;
ranking the same light curves by BLS signal-to-noise gets 2.

Intervals are 68% bootstrap intervals over 2000 paired resamples of the test
set. With 34 positives the point estimate carries ±0.07, so the third decimal
means nothing — but the model out-scores the baseline in **100.0%** of paired
resamples, so the gap does.

![Precision-recall](figures/02_precision_recall.png)

At the chosen operating threshold, on the 840 held-out light curves:

```
                    pred: no planet    pred: planet
  truth: no planet          771              35
  truth: planet              10              24

  precision 0.407    recall 0.706    F1 0.516
  false positives by true object type -> eclipsing binary: 8, variable star: 27
```

---

## Why accuracy and ROC-AUC are somewhat trivial metrics

At a 4% positive rate, the classifier `return 0` scores **96% accuracy** and
finds nothing. Any metric a constant function can win is not measuring the
thing we care about, so accuracy is not computed anywhere in this project —
`tests/test_pipeline.py` asserts the string never appears in the report.

ROC-AUC is reported, and it is not the headline either. The false-positive-rate
axis has ~800 negatives in its denominator on the test set. Going from 5 false
positives to 50 — the difference between a candidate list a telescope-time
allocation committee will fund and one it will not — moves FPR by 5 percentage
points and barely bends the curve. **Precision** has the number of *flagged*
objects in its denominator, which is the quantity that converts into nights of
follow-up, so precision-recall is the right plane and **average precision** is
the headline number.

Average precision has to be quoted against its own null: a random ranking
scores AP = P(positive) = 0.040 here. An AP of 0.73 at a 4% positive rate is an
18× lift; the same 0.73 at a 50% positive rate would be mediocre. The pipeline
always prints both.

### Choosing the threshold

The threshold is not 0.5, and it is not tuned on the test set. It is chosen by
5-fold cross-validation **inside the training split**, as the highest-recall
point whose out-of-fold precision is at least 0.50, and then frozen.

The 0.50 target comes from what the decision costs. A candidate above the
threshold buys a follow-up campaign: seeing-limited ground-based photometry to
check the event is on-target, high-resolution imaging, then radial velocities —
nights of telescope time each. A missed planet costs nothing today; it is still
in the archive and the next sector will find it. So the sensible policy is to
cap the waste rate and take whatever recall that buys. Precision ≥ 0.50 means
at most one wasted campaign per confirmed planet.

**What did not work: the target was not met out of sample.** Cross-validation on
the training split promised precision 0.500 at recall 0.726; the held-out set
delivered precision **0.407** at recall 0.706. Recall generalised; precision did
not. This is selection bias in the threshold itself — picking the operating
point that just clears a precision floor, on out-of-fold predictions from only
62 positives, is an optimisation, and the precision *at the selected point* is
biased upward by roughly the amount seen here. The fix is to select on a lower
confidence bound of the CV precision rather than its point estimate, or to
repeat the CV over several seeds and take the median. Neither is implemented;
the honest number is reported instead of a tuned one.

---

## The data

**The demo runs on synthetic photometry, because labelled real data requires
injection-recovery.** The generator is astrophysically motivated
rather than decorative — each light curve is built from the components that
appear in a real one:

```
flux = 1
     + stellar variability   sum of 3 harmonics, P ~ log-U(0.4, 25) d, amplitude ~ log-U(1e-4, 8e-3)
     + red noise             1/f^alpha, alpha ~ U(0.8, 2.0), RMS 0.4-2.5x white
     + instrumental ramps    exponential settling after momentum dumps
     + flares                sharp-rise/exponential-decay, strictly positive
     - transit or eclipse    trapezoidal, from real geometry
     + white noise           from a TESS magnitude-to-scatter relation
```

Sampling is one TESS sector: 27.4 days at 30-minute cadence, with a 1-day
mid-sector downlink gap and 3% scattered cadence losses. Each curve is ~1300
cadences.

Three populations, and the second one is the point of the exercise:

| Population | Rate | Label | Why |
|---|---|---|---|
| **Planet** | 4% | 1 | Depth `(Rp/R*)^2`, duration from Kepler's third law and the impact parameter, so depth/duration/period are *correlated the way real transits are*. A classifier cannot cheat on a combination that never occurs in nature. |
| **Eclipsing binary** | 6% | 0 | The astrophysical false positives that dominate real candidate lists. Often grazing and V-shaped, sometimes with a secondary eclipse or an odd/even depth difference. Depths overlap the planet range at the shallow end. |
| **Variable star** | 90% | 0 | Everything else: rotation, pulsation, red noise, systematics. |

Including eclipsing binaries as *labelled negatives* is what makes this a
classification problem rather than a signal-detection problem. It is why
odd/even and secondary-eclipse tests exist in real vetting pipelines, and it is
why those two features turn out to be the model's second and third most
important.

### Designed for real data to drop in

`transitml/data/base.py` defines `LightCurveSource`; nothing downstream of it
knows whether a curve was simulated or downloaded. There are two
implementations:

- `transitml/data/synthetic.py` — `SyntheticTESSSource`, used by the demo.
- `transitml/data/mast.py` — `MASTLightCurveSource`, which pulls real TESS or
  Kepler photometry from MAST through `lightkurve`. It is fully written and
  implements the same interface; it is not exercised by the demo because real
  light curves need labels from injection-recovery (see below). Switching is
  one line in `run_pipeline.py`:

```python
source = MASTLightCurveSource(
    targets=[("TIC 307210830", 1), ("TIC 100100827", 0), ...],
    mission="TESS", author="SPOC", exposure_time=1800,
)
```

The same detrending, features, model and evaluation then run unchanged. Labels
for real data are the hard part, and `mast.py` documents the two honest options
(confirmed-planet catalogues, which inherit the selection function of the
pipelines we are trying to beat; or injection-recovery into real out-of-transit
photometry, which is what the mission teams actually do).

---

## Preprocessing: the part that decides whether anything else matters

This is the crux, and it is where most of the engineering went. A detrender that
quietly eats 40% of the transit depth still produces a plausible-looking
pipeline with a quietly worse answer.

![Light curves](figures/01_light_curves.png)

The trend model is **one robust fit**:

```
trend(t) = cubic B-spline(t) + Fourier terms at the star's rotation period
```

fitted by iteratively reweighted least squares with a Tukey biweight loss, and
subtracted. The spline handles slow, non-periodic drift; the Fourier terms
handle coherent variability too fast for a safe spline to follow. They are
fitted *together*, in one design matrix, so there is no question about which
stage gets to claim which signal.

### Why not a running median

A running median is the obvious choice, and it is wrong in a way that is easy
to miss. **The median of a window over a monotone series is that window's centre
value — exactly.** On a star varying fast enough that its trend moves by more
than the photometric noise across one window, the running median therefore
reproduces the data, noise and transit included, and removing it removes the
transit.

This is not hypothetical: it is what the first version of this module did. On a
0.8% variable carrying a 3000 ppm transit, a 0.75 d running median recovers
**41%** of the injected depth. The robust spline fit recovers **101%** — unbiased
to within the measurement's own ~2% noise. Both numbers are asserted in
`test_running_median_eats_the_transit_on_a_steep_star`, which also asserts the
mechanism directly — on steep stretches the median residual is *identically
zero*.

The robust fit has no such failure mode because it never tries to pass through
the data. The transit becomes a cluster of large, one-sided residuals, the
biweight drives their weights to zero, and the trend ends up fitted to
out-of-transit cadences only. This is the "mask the transit, then fit the
baseline" recipe that Kepler and TESS pipelines apply explicitly — here the mask
is discovered rather than supplied.
`test_robust_fit_ignores_the_transit_by_zero_weighting_it` asserts that
in-transit cadences end with weight < 0.05 while out-of-transit cadences keep
weight > 0.8.

### Why the knot spacing has a floor

That protection only works while the transit *is* an outlier. Give the spline
knots tight enough to bend into a three-hour dip and the first, unweighted
iteration fits the transit; its residuals come out small; the biweight sees
nothing to reject; IRLS converges to the wrong answer. At 0.75 d knots and a
0.26 d maximum transit duration, no basis function is narrow enough to absorb
the event. `test_knot_spacing_that_is_too_tight_destroys_the_transit` sets the
knots to 0.05 d and asserts that less than 70% of the depth survives.

That floor is also why the Fourier terms exist: no safe knot spacing can follow
a 12-hour rotator, so fast rotation gets a parametric model instead. Its period
comes from a Lomb-Scargle peak — computed on *winsorised* residuals, because a
deep transit is itself a periodic signal and will otherwise define the peak of
an otherwise quiet star — and its harmonic aliases are resolved by BIC at
matched bandwidth. A spotted star is not a sinusoid: with two spot groups on
opposite hemispheres the tallest periodogram peak sits at *half* the rotation
period, and modelling the star there leaves the fundamental standing for the
transit search to lock onto. The rotation term is kept only if it shrinks the
robust residual scale by at least 5%, which is what stops a quiet star from
being given a spurious rotation model aimed at its planet.

### Clipping is upward only

Flares and cosmic rays are positive excursions; transits are negative ones. A
symmetric sigma clip — the default in most tutorials — deletes exactly the
cadences the search depends on, and deletes them hardest for the deepest, most
detectable transits. `test_clipping_is_upward_only` injects three 2% flares and
asserts that every in-transit cadence survives and the depth is still recovered
to 6%.

**Result:** injected depth is preserved to within 5% across a grid of
variability amplitudes from 0 to 0.8% and periods from 0.5 to 20 days — 20
parametrised test cases — and the detrended scatter comes back at the injected
white-noise floor.

---

## Features and model

`transitml/features.py` runs a Box Least Squares periodogram (astropy) over a
log-spaced grid of 2000 periods from 0.5 d to half the baseline, and turns the
result into 23 features in four groups:

- **Detection strength** — `bls_sde` (robust periodogram peak significance),
  `bls_depth_snr`, `bls_depth_over_scatter`, `delta_loglike`, `power_contrast`
  (peak power over the best power at an unrelated period).
- **Geometry / physical plausibility** — `log_depth`, `bls_duration`,
  `duration_over_period`, and `log_duration_ratio`: the measured duration
  against the longest duration Kepler's third law permits at that period for a
  solar-density star. An event much longer than that cannot be a planet transit,
  whatever its depth. This is the same fitted-stellar-density consistency check
  real vetting pipelines apply.
- **False-positive discriminants** — `odd_even_sigma`, `secondary_sigma`,
  `half_period_depth_ratio`, `flat_bottom_fraction` (a trapezoid fit's T23/T14,
  ~0.8 for a box and ~0 for a V), `harmonic_delta_loglike`.
- **Noise characterisation** — `red_noise_beta` (the Pont, Zucker & Queloz 2006
  beta factor: binned scatter over the white-noise expectation, so β ≫ 1 means
  the light curve has structure on exactly the timescale a transit lives on and
  the nominal depth SNR is overstated by that factor), `log_scatter`,
  `max_single_event_fraction`, `flux_skew`, `clipped_fraction`.

The classifier is `HistGradientBoostingClassifier` with shallow trees, strong
L2, `class_weight="balanced"`, and NaN handled natively — a missing odd/even
test means "too few transits to run the test", which is information, not
something to impute away.

![Feature importance](figures/04_feature_importance.png)

Permutation importance is measured in **average precision**, not accuracy, for
the same reason the headline metric is: permuting a feature barely moves
accuracy at a 4% positive rate, so an accuracy-scored importance plot would be
flat and uninformative.

### Why not a 1D CNN on folded light curves

The honest answer is sample size. This demo has 96 positives, 62 of them in the
training split. AstroNet (Shallue & Vanderburg 2018) trained on ~15,000
labelled Kepler TCEs; ExoMiner used ~35,000. A convolutional network with 10⁵–10⁶
parameters trained on 62 positives memorises them. Gradient boosting on 23
features is in the right regime for this much data, and the same argument
applies to any real single-sector study.

Two secondary reasons: the sensitivity gain in transit detection comes from
phase-folding N transits together, and BLS already does that optimally for a box
model — a CNN on unfolded curves has to rediscover folding from data, and a CNN
on folded curves needs BLS first anyway. And every feature here maps onto a test
a human vetter or the Kepler Robovetter applies, so when the model says no, the
importances say why.

**The counterpoint stands:** with 10⁵ real light curves, a CNN on global+local
folded views does beat this, because transit *shape* carries information that a
handful of scalars throws away. The choice here is a consequence of the data
volume, not a claim about architectures.

---

## Failure modes

![Diagnostics](figures/03_diagnostics.png)

Recall is not one number. A planet is missed either because the **search** never
found its period — no classifier can rescue that — or because the search found
it and the **classifier** rejected it. Those have different fixes, and reporting
only their product hides which one binds:

```
  SNR bin            n   recall   search  classifier
  0 - 7              4     0.00     0.00         n/a
  7 - 12             4     0.25     0.50        0.50
  12 - 20            1     1.00     1.00        1.00
  20 - 40            8     0.88     0.88        1.00
  40 - inf          17     0.88     1.00        0.88
```

Below SNR ~12 the search is the binding constraint and the classifier never gets
a chance. Above SNR ~40 the search is perfect and **every remaining miss is the
classifier's**. That crossover is the useful finding, and it says where effort
should go next.

**1. Low SNR.** Four test planets have injected SNR below 7; none are recovered,
and none should be. That is not a defect, it is the detection limit, and a
pipeline claiming to recover 5-sigma signals is measuring something other than
the transit.

**2. Long period / too few transits.** `SYN-002073` (P = 9.8 d, 3 transits) and
`SYN-000935` (P = 9.0 d, 3 transits) are both missed at the search stage. With a
27.4-day baseline, a 10-day planet contributes at most three events, the folding
gain is `sqrt(3)`, and the BLS peak is not distinguishable from the alias forest.
Single-transit events are excluded by construction — the period grid is capped at
half the baseline, because a single event cannot be confirmed as periodic. Real
surveys solve this by stacking sectors, not by better statistics on one.

**3. Grazing, V-shaped transits.** `SYN-001633` (b = 0.94) and `SYN-000250`
(b = 0.92) are both missed. A grazing planet produces exactly the V-shaped,
short, shallow event that `flat_bottom_fraction` and the binary tests are built
to reject. **This is a real cost of the eclipsing-binary discriminants, not a
bug**: the features that let the model beat the SNR baseline by 4× are the same
features that throw away grazing planets. Nothing in a single-sector light curve
distinguishes a grazing planet from a grazing binary; that takes radial
velocities.

**4. Red noise faking a binary signature.** The two highest-SNR misses are the
interesting ones, and the report names the reason for each:

```
SYN-001624  SNR  62  -> period recovered; classifier rejected it
                        (odd/even 3.7 sigma, secondary 2.4 sigma, red-noise beta 1.50)
SYN-001838  SNR 108  -> period recovered; classifier rejected it
                        (odd/even 5.0 sigma, secondary 4.1 sigma, red-noise beta 2.31)
```

Both are unambiguous transits that the search found perfectly. Both have
correlated noise on transit timescales (β = 1.5 and 2.3), which produced a
spurious odd/even depth difference and a spurious secondary eclipse, and the
model correctly applied binary logic to a false premise. The right fix is to
compare the odd/even and secondary statistics against the *empirical* noise at
that timescale — divide them by β — rather than against the white-noise error
bars astropy returns. That is a concrete next step, not a hand-wave.

**5. False positives are mostly variable stars, not binaries.** Of 35 false
positives, 8 are eclipsing binaries and 27 are plain variable stars. Per object
that is a **15.1%** false-positive rate on the 53 binaries against **3.6%** on
the 753 variable stars — binaries are four times more likely to fool the model,
exactly as expected, but residual variability that survives detrending still
dominates the candidate list by sheer weight of numbers. That matches the real
TESS experience, where most rejected candidates are systematics rather than
astrophysical false positives.

---

## What this does and does not prove

**What it establishes.** The pipeline is real and the numbers are real: the
detrender provably preserves transit depth to 5% while removing variability ten
times deeper; the search recovers every held-out injection above SNR 40 and 88%
above SNR 20; the classifier beats a strong single-statistic baseline by 4× in
average precision on data it has never seen — in 100% of paired bootstrap
resamples — with the threshold frozen from training-split cross-validation. The evaluation protocol — average precision against the
positive-rate null, threshold from CV, recall decomposed into search and
classifier — is exactly what one would use on real data, and
`tests/test_evaluation.py` asserts that corrupting the training rows changes no
reported number while corrupting the test rows does.

**What it does not establish.** The noise model is the weak point, and it is
weak in the direction that flatters the result. Real TESS systematics are
*structured* — scattered light from the Earth and Moon on a 13.7-day orbital
cycle, focus changes with spacecraft thermal state, pointing jitter correlated
across a whole camera, background contamination from neighbouring stars in a
21-arcsecond pixel — and none of that is a stationary 1/f process. My red noise
is generated as a power law with random phases, so it has no features the
detrending can fail on in a correlated way across targets, and a robust spline
handles it more easily than it would handle real data. I would expect average
precision to drop substantially on real photometry, and the drop to come mostly
from the false-positive side.

Three further gaps:

- **Blends.** The single largest astrophysical false-positive class in real TESS
  data is a background eclipsing binary diluted by a bright foreground star
  inside the same pixel. It looks exactly like a shallow planet transit and is
  separated by *centroid motion* — the flux-weighted centroid shifts during the
  event — which requires pixel-level data this pipeline never sees. Every real
  vetting system uses centroid tests; none are here, and adding them would need
  target pixel files rather than light curves.
- **Labels.** Ground truth is known by construction here. On real data it has to
  come from a catalogue that inherits the selection function of the pipelines
  being benchmarked against, or from injection-recovery, which only measures
  completeness and not the false-positive rate.
- **Sample size.** 96 positives in total and 34 in the test set. The bootstrap
  interval on average precision is [0.659, 0.799] — ±0.07 — so the difference
  between 0.73 and 0.70 is noise, and only the gap to the baselines is
  meaningful. The pipeline reports the interval so this cannot be over-read.

The honest summary: this demonstrates the *method* — correct detrending, correct
features, correct metric, correct protocol, honest failure analysis — on data
whose noise is easier than reality. The next step on real data is
injection-recovery into genuine TESS out-of-transit photometry, which keeps the
systematics real while keeping the labels trustworthy.

---

## Layout

```
transit-detection/
├── run_pipeline.py             # the one command
├── requirements.txt            # pinned
├── pyproject.toml              # package metadata + pytest config
├── transitml/
│   ├── config.py               # every tunable number, in one dataclass tree
│   ├── physics.py              # Kepler's third law, transit durations
│   ├── data/
│   │   ├── base.py             # LightCurve + LightCurveSource interface
│   │   ├── synthetic.py        # the generator
│   │   ├── mast.py             # real TESS/Kepler via lightkurve (same interface)
│   │   └── loader.py           # source -> feature matrix, parallel over curves
│   ├── preprocess.py           # robust spline + rotation detrending
│   ├── features.py             # BLS search and vetting statistics
│   ├── model.py                # split, baselines, training, threshold selection
│   ├── evaluate.py             # PR curves, AP, confusion matrix, failure analysis
│   └── plots.py                # figures (matplotlib Agg, no display)
├── tests/                      # 86 tests, ~1 min
│   ├── test_generator.py       # imbalance is exact; injected physics is consistent
│   ├── test_preprocess.py      # depth preservation; why the median was rejected
│   ├── test_features.py        # recovery vs SNR; the vetting statistics fire
│   ├── test_evaluation.py      # no test-set leakage into threshold or metrics
│   └── test_pipeline.py        # end to end, reproducible, figures on disk
├── figures/                    # committed, so this README renders
└── results/                    # metrics.json + report.txt, committed
```

`python run_pipeline.py --help` exposes `--seed`, `--n-curves`, `--n-jobs`,
`--no-figures` and the output directories. Runtime scales linearly in
`--n-curves`; the BLS search is the bottleneck and is parallel across curves.

## References

- Kovács, Zucker & Mazeh (2002) — Box Least Squares.
- Pont, Zucker & Queloz (2006) — the red-noise β factor.
- Vanderburg & Johnson (2014) — robust spline detrending with iterative outlier
  rejection.
- Shallue & Vanderburg (2018) — AstroNet; the CNN comparison point.
- Astropy `BoxLeastSquares` and `LombScargle` implementations.
