# Transit detection in TESS-like light curves

Finding transiting exoplanets is a needle-in-a-haystack detection problem: a few
per cent of stars host a detectable transiting planet, the signal is a 0.05–1%
dip lasting a few hours, and it sits on top of stellar variability ten to a
hundred times deeper. This repository is an end-to-end pipeline for that
problem — generate the photometry, detrend it, search it, classify it, and
evaluate it the way an imbalanced detection problem has to be evaluated.
It also vets single real stars: `python -m transitml.vet "TIC ..."` downloads,
detrends, searches and scores one target and writes a one-page report.

**One command reproduces everything in this README:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py            # ~2 min on 4 cores
pytest                            # ~3 min, 270 tests
```

It writes `results/metrics.json`, `results/report.txt`, the trained model
(`results/model.joblib`, used by `vet` below) and four PNGs to `figures/`.
Fixed seed (42), pinned dependencies, deterministic output.

---

## Headline result

2400 light curves, 4.0% of them hosting a transiting planet, 35% held out.

| Scorer | Average precision (68% CI) | ROC-AUC | Precision@20 |
|---|---|---|---|
| Random ranking (chance) | 0.040 | 0.500 | — |
| Baseline: rank by BLS **depth** | 0.097 [0.080, 0.128] | 0.751 | 0.00 |
| Baseline: rank by BLS **depth SNR** | 0.180 [0.149, 0.231] | 0.830 | 0.10 |
| **Gradient boosting on vetting features** | **0.799** [0.732, 0.862] | 0.939 | **1.00** |

**20× better than chance, 4.4× better than the strongest single-statistic
baseline.** Of the 20 candidates the model ranks highest, all 20 are real planets;
ranking the same light curves by BLS signal-to-noise gets 2.

Intervals are 68% bootstrap intervals over 2000 paired resamples of the test
set. With 34 positives the point estimate carries ±0.07, so the third decimal
means nothing — but the model out-scores the baseline in **100.0%** of paired
resamples, so the gap does.

![Precision-recall](figures/02_precision_recall.png)

At the chosen operating threshold, on the 840 held-out light curves:

```
                    pred: no planet    pred: planet
  truth: no planet          778              28
  truth: planet               8              26

  precision 0.481    recall 0.765    F1 0.591
  false positives by true object type -> eclipsing binary: 8, variable star: 20
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
scores AP = P(positive) = 0.040 here. An AP of 0.80 at a 4% positive rate is a
20× lift; the same 0.80 at a 50% positive rate would be mediocre. The pipeline
always prints both.

### Choosing the threshold

The threshold is not 0.5, and it is not tuned on the test set. It is chosen by
5-fold cross-validation **inside the training split**, as the highest-recall
point whose out-of-fold precision is at least 0.50 *at a one-sided 1-sigma
Wilson lower confidence bound*, and then frozen.

The 0.50 target comes from what the decision costs. A candidate above the
threshold buys a follow-up campaign: seeing-limited ground-based photometry to
check the event is on-target, high-resolution imaging, then radial velocities —
nights of telescope time each. A missed planet costs nothing today; it is still
in the archive and the next sector will find it. So the sensible policy is to
cap the waste rate and take whatever recall that buys. Precision ≥ 0.50 means
at most one wasted campaign per confirmed planet.

**What did not work at first, and what was done about it.** The first version
applied the 0.50 floor to the *point estimate* of out-of-fold precision.
Cross-validation promised precision 0.500 at recall 0.726; the held-out set
delivered **0.407** at recall 0.706. That gap is selection bias in the threshold
itself: taking the deepest point that just clears a precision floor, on
out-of-fold predictions from only 62 positives, picks a point whose precision
happened to fluctuate upward.

The rule now requires a one-sided Wilson score lower bound on the CV precision
(TP over TP + FP at each candidate threshold, `precision_lcb_z = 1.0` in
`EvalConfig`, i.e. one sigma, matching the 68% intervals used everywhere else)
to clear 0.50. The bound never exceeds the point estimate, so this can only
raise the threshold, and `tests/test_evaluation.py` asserts that on random and
real out-of-fold scores. On this run the bound is 0.506 at 82 candidates, the
CV point estimate there is 0.561 at recall 0.742, and the held-out set delivers
precision **0.481** at recall 0.765.

So the correction helped but did not close the gap: held-out precision is still
below the 0.50 target, by less than the 68% sampling uncertainty of a 54-candidate
test sample (about ±0.07), but below it nonetheless. Separating the two changes
made in the same round, with the beta-scaled features (next sections) and the
old point-estimate rule the threshold would have been 0.167 and held-out
precision 0.426 (35 false positives); the Wilson rule moves it to 0.200 and 28
false positives at unchanged recall. A one-sigma bound on 62 positives is a
modest correction, and the CV-to-test drop (0.561 to 0.481) shows the bias is
larger than one sigma here. Raising `precision_lcb_z` or taking the median over
several CV seeds are the obvious next steps; neither was tried, because tuning
the knob after seeing the test number would reintroduce exactly the bias it is
meant to remove.

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
  Kepler photometry from MAST through `lightkurve`. It implements the same
  interface. The synthetic demo does not use it, because real light curves
  need labels; the real-photometry injection run below and the `vet` command
  do. Pointing the pipeline at it directly looks like this:

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

### Injection-recovery on real photometry

The second option is implemented. `transitml/data/injection.py` takes real
light curves of stars with no known planet and injects the same planet and
eclipsing-binary population the synthetic generator draws, from the same
functions, at the same exact class rates. Uninjected curves come back
byte-for-byte untouched, so every curve carries real TESS noise and only the
labels are constructed. Eclipses are multiplied into the flux, so aperture
contamination dilutes them as it would a real event.

```bash
pip install lightkurve
python run_pipeline.py --inject-into targets.txt --exclude-tois toi.csv --sector 14
```

- `targets.txt` is a list of TIC IDs (one per line, or a CSV with a `TIC ID`
  column), for example the TESS-SPOC full-frame-image targets of one sector.
- `toi.csv` is the TOI table exported from ExoFOP. Every listed TIC is
  dropped, because a known host left in would become a mislabelled negative.
  Hosts of planets nobody has found yet cannot be excluded; at a few per cent
  of stars they bias precision slightly downward, which is the safe direction.
- One sector per star is used, so a star cannot appear in both splits.
- Downloaded curves are cached to `results/real_injection/base_curves.npz`;
  later runs read the cache and need no network, so a real-data result is as
  reproducible as the synthetic one. `--n-curves` caps the target list.
- Results and figures go to `results/real_injection/` and
  `figures/real_injection/`, never over the synthetic headline.

**Result on real photometry (TESS sector 14).** 2,800 TESS-SPOC 30-minute
FFI curves, drawn at random from the sector's target list after removing every
TOI host (`data/real_injection/`), with 112 planets and 168 eclipsing binaries
injected. Held-out set: 980 curves, 39 planets. Full report in
[`results/real_injection/report.txt`](results/real_injection/report.txt).

| | Synthetic | Real sector 14 |
|---|---|---|
| Model average precision | 0.80 | **0.51** [0.44, 0.60] |
| Best baseline (BLS SNR) | | 0.10 |
| Held-out precision / recall | | 0.48 / 0.54 |
| Precision of top 20 | | 0.75 |

The prediction held: real noise costs a lot of average precision. Most of
the loss is at low SNR, where the search itself stops finding the period
(30% recovered below SNR 7, 83% at SNR 7 to 12, 16 of 17 above 12), and on the
false-positive side, where 15 of the 23 false positives at the operating
point are real stars with nothing injected rather than binaries.

To reproduce (the TOI table is the NASA Exoplanet Archive `toi` table, the
same list ExoFOP serves):

```bash
python run_pipeline.py --inject-into data/real_injection/targets_s0014.txt \
    --exclude-tois data/real_injection/toi.csv --sector 14 --download-workers 16
```

### Benchmark against real labels: TOI dispositions

Injection-recovery keeps the noise real but still draws the signals from this
project's own planet and binary models. The other honest test is the one a
vetting tool faces in use: real TESS signals that follow-up observers have
since resolved. `--benchmark-tois` takes the ExoFOP TOI table, labels each star
by its TFOPWG disposition, and scores the trained model on it **unchanged**,
threshold included:

| Disposition | Label |
|---|---|
| CP (confirmed planet), KP (known planet) | 1 |
| FP (false positive), FA (false alarm) | 0 |
| PC, APC (still open) | left out |

A star is a planet host if any of its TOIs is CP or KP, and a negative only if
every TOI on it is FP or FA. Each star is scored on one sector (the first of
`--benchmark-sectors` it was observed in), and every star the model was
trained on is removed first. Code in `transitml/data/toi.py` and
`transitml/benchmark.py`; the table used is a 2026-10-06 ExoFOP snapshot in
`data/toi_benchmark/`.

Two things make these numbers different in kind from the ones above. Every
star here is a TOI, so **both classes already passed a TESS pipeline's
detection and automated vetting**: the false positives are the hard ones that
got through, many of them eclipsing binaries on or near the target, not
random variable stars. And the catalogue, not the sky, sets the positive rate
(about 50% here), so precision does not transfer to a survey. The two numbers that do
transfer are **recall on confirmed planets** and **the fraction of known false
positives rejected** at the frozen threshold.

**Result (TESS sectors 14 to 26, 30-minute TESS-SPOC curves).** 1,003 labelled
TOI hosts were observed in those sectors; MAST has a TESS-SPOC light curve for
746 of them (376 CP/KP, 370 FP/FA). Full reports:
[`results/real_injection/toi_benchmark.txt`](results/real_injection/toi_benchmark.txt)
and [`results/toi_benchmark.txt`](results/toi_benchmark.txt).

| Model trained on | Planets kept | False positives rejected | AP (chance 0.50) | ROC-AUC | Top 20 |
|---|---|---|---|---|---|
| Injections into real sector 14 noise | 0.55 | 0.62 | **0.62** [0.59, 0.64] | 0.59 | 0.85 |
| Synthetic light curves | 0.57 | 0.59 | 0.61 [0.58, 0.63] | 0.59 | 0.65 |
| Baseline: rank by BLS SNR | | | 0.60 [0.57, 0.63] | 0.58 | 0.65 |

![TOI benchmark](figures/real_injection/05_toi_benchmark.png)

**The model barely separates confirmed planets from TOI false positives.**
It beats ranking by BLS signal-to-noise in only 76% of paired bootstrap
resamples, and the table of what it keeps by catalogued depth shows why: the
fraction of planets kept and the fraction of false positives kept rise
together, from 0.28 and 0.16 below 1000 ppm to 0.77 and 0.71 at 6000 to 10000
ppm. On this population the model is mostly a signal-strength ranking. Its
top 20 are 85% real planets (65% for the synthetic-trained model and for the
BLS SNR ranking), so the most confident end of the ranking is still useful.
Training on real noise rather than synthetic noise makes no difference here
that the intervals can resolve.

Two causes are visible in the report, and they point at the next work:

- **The odd/even test fails at high SNR on real data.** The confirmed planets
  the model rejects with the most confidence are bright hot Jupiters (TOI
  1151.01 at SNR 1017, TOI 1431.01, TOI 1682.01), and they carry odd/even
  depth differences of 8 to 37 sigma even after the red-noise beta correction.
  At SNR in the hundreds, a fraction of a per cent of systematic depth
  difference between alternate transits is many sigma. The training data had
  no such systematics, so the model learned that a large odd/even sigma means
  a binary. A floor on the odd/even and secondary uncertainty proportional to
  the depth would fix this, and is a change to `features.py`.
- **Blended binaries look like planets in a light curve.** The false positives
  the model keeps have median odd/even and secondary significances of 0.8 and
  0.2 sigma, indistinguishable from the planets'. That is what a background
  eclipsing binary diluted by a brighter neighbour looks like.
  Separating them needs pixel data. `vet --centroids` now runs a centroid
  test from target pixel files (see "Centroid test" below), but the model and
  this benchmark see light curves only.

The search is the other ceiling: BLS recovers the catalogued period for 73% of
planets in one sector (88% above TOI SNR 40, 32% below 10), and when it
misses the period the planet is kept 8% of the time.

To reproduce (downloads about 750 curves the first time and caches them to
`toi_curves.npz` beside the results):

```bash
python run_pipeline.py --inject-into data/real_injection/targets_s0014.txt \
    --exclude-tois data/real_injection/toi.csv --sector 14 \
    --benchmark-tois data/toi_benchmark/exofop_toi_2026-10-06.csv --benchmark-sectors 14-26
```

The same flags work after the synthetic run (drop `--inject-into` and
`--exclude-tois`). A fresh table comes from
`https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv`.

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
  The two significances are divided by the red-noise β (below, floored at 1)
  measured with both the primary and the phase-0.5 windows masked, so they are
  judged against the empirical noise on the transit timescale rather than
  white-noise error bars.
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

## Vetting one star

`python -m transitml.vet` scores a single light curve with the model
`run_pipeline.py` saved, and explains the score on one page.

```bash
python run_pipeline.py                                   # writes results/model.joblib
python -m transitml.vet star.csv                         # columns time,flux[,flux_err]
python -m transitml.vet results/real_injection/base_curves.npz --target-id "TIC 123"
python -m transitml.vet "TIC 307210830" --sector 14      # needs lightkurve and network
python -m transitml.vet "TIC 307210830" --stitch         # every sector, joined
```

It detrends and featurises with the preprocessing and BLS settings stored in
the model file, so the score means what it meant in training, then writes
`results/vet/vet_<target>.png` and a JSON of the same numbers: the raw curve
and removed trend, the detrended curve with every candidate signal marked,
the primary fold, odd against even transits, the phase-0.5 window, the key
features with score, threshold and verdict, and the top reasons. The reasons
are approximate and labelled so: each is the change in score when one
feature alone is set to its training-split median. They are not additive.

**The TIC path has only been tested offline**, with the MAST source replaced
by a stub; this environment could not reach MAST. Local CSV and npz input is
tested end to end. The model file is a pickle tied to the scikit-learn
version that wrote it, so it is git-ignored and rebuilt by `run_pipeline.py`;
load only model files you made.

**More than one planet.** The classifier scores only the strongest BLS peak,
and the headline numbers are unchanged. `transitml/search.py` adds an
iterative search for the vetting report: find the best signal, mask 1.5
durations around each of its transits, search again, up to three signals,
stopping at the first peak whose `bls_sde` is below 5.5. On the synthetic run
3.7% of plain variable stars clear 5.5 on their first search, against 62% of
planets. A light curve with nothing significant yields an empty list.

**More than one sector.** `stitch_light_curves` (in `transitml/data/base.py`)
joins sectors of one star: each is divided by its own median, then they are
concatenated with the gaps left as gaps. `MASTLightCurveSource(...,
stitch_sectors=True)` yields one stitched curve per target; the default is
still one curve per sector. Detrending already splits on gaps, so no spline
segment spans one, and the BLS grid extends to half the stitched baseline,
which brings the long-period planets of failure mode 2 below within reach.
The test that pins this uses a 10-day planet with three transits per sector:
neither sector alone gives a significant peak at 10 days, the two stitched
together do. The grid keeps its 2000 periods however long the baseline, so
sectors far apart in time are searched too coarsely; consecutive sectors
are fine.

### Centroid test: is the dip on the target?

The largest astrophysical false-positive class in real TESS data is a
background eclipsing binary a pixel or two from the target, diluted by the
target's light into a shallow, planet-like dip. The light curve cannot tell
the two apart; the pixels can, because the missing light goes missing from
where its source is. `vet` takes target pixel files for this:

```bash
python -m transitml.vet star.csv --tpf star_tpf.npz               # saved pixels
python -m transitml.vet "TIC 307210830" --sector 14 --centroids   # downloads them
```

`--tpf` reads a file written by `transitml.data.tpf.save_tpf` (plain arrays,
no pickle; repeat the flag for several sectors). `--centroids` downloads the
star's target pixel files through lightkurve with the light curve's author,
cadence and sector. The pixel times must be on the light curve's time system
(BTJD for anything from MAST). On the primary signal's ephemeris,
`transitml/centroid.py` runs two tests, after the Kepler and TESS
data-validation reports:

- **Difference image.** For each transit, the mean image of the cadences
  just before and after it (a quarter-duration gap, then 1.5 durations on
  each side) minus the mean in-transit image, averaged over transits. Its
  flux-weighted centroid is where the dip happens. It is compared with the
  target's catalogue position from the file's WCS, and also reported against
  the out-of-transit centroid. The catalogue position is the reference for
  the flag because the out-of-transit centroid of a crowded stamp is the
  light-weighted mean of every star in it, so an on-target transit sits off
  it too; a test asserts exactly that. Without a catalogue position the
  out-of-transit centroid is used, and that bias applies.
- **Centroid motion.** The flux-weighted centroid of each cadence, in transit
  against the flanking baseline, beside the shift an on-target transit of
  the measured depth would cause. Reported, not used for the flag.

Centroids are measured over the pixels within 3 pixels of the reference, so
noise in far pixels with long lever arms does not dominate. Errors come from
a bootstrap over transits when there are at least 5, or over cadences within
transits when there are fewer (labelled as such, and optimistic when the
noise is correlated). The covariance of a handful of transits is itself
uncertain, so the offset's Mahalanobis distance is referred to Hotelling's
T-squared distribution, `F(2, n - 2)`, with `n` the number of transits or
the Welch-Satterthwaite effective sample size of the cadence bootstrap, and
quoted as a Gaussian-equivalent sigma. Treating the bootstrap covariance as
exact was badly miscalibrated: a nominal 3-sigma offset was reached by 7% of
on-target transits with 9 transits and by 43% with 3. An offset is flagged
when the dip is detected in the difference image (SNR at least 3), the
offset is at least 3 sigma, and it is at least 0.1 pixel (2 arcsec; TESS
pixels are 21 arcsec), a floor for what the bootstrap cannot see: an
undersampled, asymmetric real PRF and catalogue and WCS errors.

On synthetic stamps (`transitml/data/synthetic_tpf.py`: Gaussian stars
integrated over pixels, photon, sky and read noise, 0.005-pixel pointing
jitter), a T = 10 target with a neighbour 2 magnitudes fainter and 1.8
pixels away, 150 scenes each:

| Scenario | Transits | Flagged | Above 2 sigma |
|---|---|---|---|
| Planet on the target | 9 | 0% | 6% (nominal 4.6%) |
| Planet on the target | 1 to 3 | 0 to 0.7% | 4.7 to 5.3% |
| Binary on the neighbour (0.6% dip in the aperture) | 9 | 100% | 100% |
| Binary on the neighbour | 5 | 65% | 100% |
| Binary on the neighbour | 1 to 3 | 81 to 100% | 99 to 100% |

Five transits is the weakest case: the transit bootstrap is in use but an
`F(2, 3)` reference has heavy tails. Pointing jitter is the main nuisance:
at 0.03 pixel per cadence the blend is flagged in 11% of scenes, because a
bright star moving by a fraction of a pixel changes every pixel by more than
a 0.6% dip does. The flux-weighted centroid of an off-centre source is also
pulled towards the window's centre (a neighbour 1.8 pixels away measures at
1.5 to 1.7), so the direction and the significance are what to read and the
length is a lower bound.

The result goes into the JSON as a `centroid` section (`offset_flag`, one
entry per file with every number above, and a status such as
`no_in_transit_cadences` or `weak_difference_image` when the test cannot
be run or is not meaningful) and into the PNG as a fifth row: the
out-of-transit image with the aperture outlined, the difference image, and
the numbers. A flagged offset is also called out under the title and printed.
**The score is unchanged**: the model was never trained on centroid
features, so the centroid result is a separate vetting test reported beside
it, not folded into it.

**The download has only been tested offline**, with lightkurve replaced by
stand-ins, as for the light curves: this environment could not reach MAST.
The conversion reads the pipeline aperture (falling back to lightkurve's
threshold mask), the CCD origin, and the target position from the WCS.
Nothing has been run on a real target pixel file, and the synthetic PSF is a
circular Gaussian, much tidier than the TESS PRF. The test also uses only the
primary signal and one ephemeris per run, and is not applied in the
`run_pipeline.py` evaluation, which has no pixels.

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
  40 - inf          17     1.00     1.00        1.00
```

Below SNR ~12 the search is the binding constraint and the classifier never gets
a chance. Above SNR ~40 both stages are now perfect on the held-out set. In the
first version they were not: the search found all 17 planets there, but the
classifier rejected two of them (failure mode 4 below), which is what pointed
at the classifier's binary tests as the place to put effort. Of the 8 planets
still missed, 7 are search misses and 1 is a classifier rejection.

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
bug**: the features that let the model beat the SNR baseline by 4.4× are the same
features that throw away grazing planets. Nothing in a single-sector light curve
distinguishes a grazing planet from a grazing binary; that takes radial
velocities.

**4. Red noise faking a binary signature (fixed).** In the first version the
two highest-SNR misses were planets the search found perfectly and the
classifier rejected on binary grounds:

```
SYN-001624  SNR  62  -> period recovered; classifier rejected it
                        (odd/even 3.7 sigma, secondary 2.4 sigma, red-noise beta 1.50)
SYN-001838  SNR 108  -> period recovered; classifier rejected it
                        (odd/even 5.0 sigma, secondary 4.1 sigma, red-noise beta 2.31)
```

Both stars have correlated noise on transit timescales, which produced a
spurious odd/even depth difference and a spurious secondary eclipse measured
against astropy's white-noise error bars. The odd/even and secondary
statistics are now divided by β, floored at 1 (Pont, Zucker & Queloz 2006),
which inflates their error bars to the empirical noise at that timescale.
Their odd/even significances drop to 2.4 and 2.2 sigma and their secondaries
to 1.6 and 1.8 sigma; the model scores them 0.99 and 0.97 and **both are now
recovered**. Held-out average precision rose from 0.730 to 0.799, and the
classifier now keeps every planet the search finds above SNR 12. Eclipsing
binary false positives did not rise (8 before and after).

One detail mattered. The `red_noise_beta` *feature* masks only the exact BLS
box, and with that mask the ingress and egress cadences of a slightly
mis-fitted period or duration leak into the out-of-transit bins; a single
leaked cadence of a deep event dominates the binned scatter. On a pure
white-noise light curve with a 3000 ppm transit it reads β = 1.5 to 2.4, so
scaling by it would have discounted the binary tests of every deep event, real
binaries included. The β used for scaling therefore masks two box-widths
around both the primary and phase 0.5 (the latter so a real secondary eclipse
cannot inflate the β that is then used to discount it), and returns β ≈ 1 on
that same light curve. `tests/test_features.py` asserts both behaviours. The
feature itself was left unchanged, so the β values printed in the report are
the feature's; for these two planets the two estimates agree.

**5. False positives are mostly variable stars, not binaries.** Of 28 false
positives, 8 are eclipsing binaries and 20 are plain variable stars. Per object
that is a **15.1%** false-positive rate on the 53 binaries against **2.7%** on
the 753 variable stars. Binaries are more than five times more likely to fool the model,
exactly as expected, but residual variability that survives detrending still
dominates the candidate list by sheer weight of numbers. That matches the real
TESS experience, where most rejected candidates are systematics rather than
astrophysical false positives.

---

## What this does and does not prove

**What it establishes.** The pipeline is real and the numbers are real: the
detrender provably preserves transit depth to 5% while removing variability ten
times deeper; the search recovers every held-out injection above SNR 40 and 88%
above SNR 20; the classifier beats a strong single-statistic baseline by 4.4× in
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
  separated by *centroid motion*: the flux-weighted centroid shifts during the
  event, which takes pixel-level data to see. The classifier and the headline
  numbers still see light curves only. `vet` can now run a difference-image
  centroid test from target pixel files beside the score (see "Centroid test"
  above), but it has been checked only on synthetic pixels with a Gaussian
  PSF, not on real TESS data, and it is not part of the evaluation.
- **Labels.** Ground truth is known by construction here. On real data it has to
  come from a catalogue that inherits the selection function of the pipelines
  being benchmarked against, or from injection-recovery, which only measures
  completeness and not the false-positive rate. The TOI benchmark above does
  the first, and shows the model separates real planets from real TOI false
  positives only slightly better than a signal-to-noise ranking.
- **Sample size.** 96 positives in total and 34 in the test set. The bootstrap
  interval on average precision is [0.732, 0.862], roughly ±0.065, so the difference
  between 0.80 and 0.77 is noise, and only the gap to the baselines is
  meaningful. The pipeline reports the interval so this cannot be over-read.

The honest summary: this demonstrates the *method* — correct detrending, correct
features, correct metric, correct protocol, honest failure analysis — on data
whose noise is easier than reality. Injection-recovery into genuine TESS
photometry, which keeps the systematics real while keeping the labels
trustworthy, has now been run on sector 14 (see "Injection-recovery on real
photometry" above), and it confirmed the prediction: average precision fell
from 0.80 to 0.51, with most false positives coming from real stars that had
nothing injected.

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
│   │   ├── base.py             # LightCurve + LightCurveSource interface, stitching
│   │   ├── synthetic.py        # the generator
│   │   ├── mast.py             # real TESS/Kepler via lightkurve (same interface)
│   │   ├── injection.py        # synthetic eclipses injected into real curves
│   │   ├── toi.py              # TOI table -> per-star CP/KP vs FP/FA labels
│   │   ├── files.py            # CSV and npz light-curve files
│   │   ├── tpf.py              # target pixel files: container, npz, lightkurve
│   │   ├── synthetic_tpf.py    # synthetic pixels: on-target transits and blends
│   │   └── loader.py           # source -> feature matrix, parallel over curves
│   ├── preprocess.py           # robust spline + rotation detrending
│   ├── features.py             # BLS search and vetting statistics
│   ├── search.py               # iterative multi-planet search
│   ├── centroid.py             # difference-image and centroid-motion tests
│   ├── model.py                # split, baselines, training, threshold, save/load
│   ├── evaluate.py             # PR curves, AP, confusion matrix, failure analysis
│   ├── benchmark.py            # the trained model scored on real TOI dispositions
│   ├── vet.py                  # python -m transitml.vet: one star, one page
│   └── plots.py                # figures (matplotlib Agg, no display)
├── tests/                      # 270 tests, ~3 min
│   ├── test_generator.py       # imbalance is exact; injected physics is consistent
│   ├── test_preprocess.py      # depth preservation; why the median was rejected
│   ├── test_features.py        # recovery vs SNR; the vetting statistics fire
│   ├── test_search.py          # two planets found; noise yields nothing
│   ├── test_stitch.py          # sectors joined; marginal pair becomes a detection
│   ├── test_evaluation.py      # no test-set leakage into threshold or metrics
│   ├── test_injection.py       # injection is exact, leaves noise alone, runs offline
│   ├── test_toi.py             # TOI parsing, per-star labels, sector choice
│   ├── test_benchmark.py       # the real-label benchmark, offline, from a cache
│   ├── test_files.py           # CSV and npz input
│   ├── test_model_io.py        # saved model reloads with threshold and features
│   ├── test_vet.py             # vet end to end on CSV, npz and a stubbed TIC
│   ├── test_tpf.py             # pixel files: npz round trip, stubbed download
│   ├── test_synthetic_tpf.py   # synthetic pixels put the light where it belongs
│   ├── test_centroid.py        # blends flagged, on-target not; bad input survives
│   ├── test_vet_centroid.py    # centroid section in JSON and PNG; score unchanged
│   └── test_pipeline.py        # end to end, reproducible, figures on disk
├── figures/                    # committed, so this README renders
└── results/                    # metrics.json + report.txt, committed
```

`python run_pipeline.py --help` exposes `--seed`, `--n-curves`, `--n-jobs`,
`--no-figures` and the output directories. Runtime scales linearly in
`--n-curves`; the BLS search is the bottleneck and is parallel across curves.

## Roadmap

What is built and what is planned, roughly in the order it is being worked
on. Sizes are rough: S is a few hours, M a day or two, L longer.

**Done**

- Odd/even and secondary-eclipse significances divided by the red-noise β.
- Operating threshold chosen on a Wilson lower bound of CV precision.
- Injection-recovery on real TESS photometry (sector 14, AP 0.51).
- `python -m transitml.vet`: one star in, a one-page vetting report out.
- Iterative multi-planet search for the vetting report.
- Multi-sector stitching (`--stitch`, `stitch_light_curves`).
- Centroid vetting from target pixel files (`vet --tpf`, `--centroids`):
  a difference-image offset and centroid motion, reported beside the score.
  Checked on synthetic pixels only so far.
- Benchmark against real TOI dispositions (`--benchmark-tois`; 746 hosts in
  sectors 14 to 26, AP 0.62 against a chance level of 0.50).

**Planned**

| Item | What it adds | Size |
| --- | --- | --- |
| Structured systematics in the generator | 13.7-day scattered light, camera-correlated jitter and focus drift, so the synthetic noise stops flattering the result | M |
| Single-transit and duo-transit search | Events the period grid excludes by construction today | M |
| Transit Least Squares and GPU BLS | An alternative search and a faster one; BLS is the runtime bottleneck | M |
| Kepler DR25 training set, then an optional CNN | About 34k labels, enough to train on transit shape | L |
| Probability calibration and per-candidate SHAP | A calibrated score and an exact reason per object | S |
| Batch mode over a whole sector | On-disk caching and a candidate list | M |
| Planet parameter fits | batman and emcee fits for candidates that pass | M |

## References

- Kovács, Zucker & Mazeh (2002) — Box Least Squares.
- Pont, Zucker & Queloz (2006) — the red-noise β factor.
- Vanderburg & Johnson (2014) — robust spline detrending with iterative outlier
  rejection.
- Shallue & Vanderburg (2018) — AstroNet; the CNN comparison point.
- Bryson et al. (2013): difference-image centroid offsets for Kepler false
  positives.
- Twicken et al. (2018): the Kepler data validation tests, including the
  centroid tests.
- Astropy `BoxLeastSquares` and `LombScargle` implementations.
