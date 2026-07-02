# gratings

Electrode grating design and analysis tools for periodic poling, quasi-phase-matching, or bragg gratings in nonlinear optical crystals and waveguides. Constructs, transforms, and analyzes electrode poling patterns suitable for generating and modeling lithography mask layouts.

A grating is a pair of lists, `starts, ends`, the positions of each poled bar, in unspecified units but µm is typical. Bar boundaries are computed analytically except where the algorithm is inherently rasterized or discrete so the output is exact to float precision.

## Install

```
git clone https://github.com/decoherer/gratings
```

Requires `numpy`, `scipy` (special functions and root finding in a few generators), and the companion [`wavedata`](https://github.com/decoherer/wavedata) package (`Wave` class, used for spectra and plotting). [`sellmeier`](https://github.com/decoherer/sellmeier) may also be helpful for choosing target poling periods.

## Quick start

```python
from gratings import Grating, Uniformgrating

g = Uniformgrating(6.9, length=10000)  # 10 mm uniform grating, Λ = 6.9 µm, 50% duty cycle
f = g.ftgrating(dp=0.1)     # QPM tuning curve: Wave of relative response vs Λ (µm)
w = g.grating2wave()        # poling pattern as a plottable square wave
g.bars2file('mask')         # write domain boundaries to mask.dat
```

Transforms return new instances and chain left to right:

```python
gg = ( Uniformgrating(30, length=10000)
       .apodizebars(apodize='asingauss', σ=0.23)  # Gaussian amplitude apodization via duty cycle
       .dropsmallbars(tolerance=0)
       .mergetouchingbars(tolerance=0)
       .shrinkbars(0.6)                           # overpole precompensation
       .dropsmallbars() )                         # enforce lithographic minimum feature
```

## The `Grating` class

```python
g = Grating(starts, ends)               # validates starts[n] < ends[n] < starts[n+1]
g = Grating(starts, ends, length=10000) # device length, stored and inherited; g.isinbounds() tests 0 ≤ bars ≤ length
starts, ends = g                        # unpacks, so f(*g) works with every module function
g.starts, g.ends, g.bars                # bars = flat [s0,e0,s1,e1,...]
len(g), g[i], g[i:j]                    # bar count, i-th bar as (start,end), slice as Grating
g.period                                # mean period estimate
g & h, g | h                            # boolean AND/OR of two binary gratings
```

`Uniformgrating(period, dc=0.5, length=..., ...)` subclasses `Grating`, wrapping `grating()`: construction parameters are kept as attributes (`g.Λ`, `g.dc`, ...), `length` is always set, and `period` returns the exact design period rather than the base class's estimate from the bars. Transforms return plain `Grating`s, since a transformed uniform grating is no longer parametrically uniform.

`length` is the device (pad) length and is inherited through every transform. Bars outside `[0, length]` are constructible — unconventional rather than invalid — and `g.isinbounds()` tests for them (e.g. duty-cycle apodization widens edge bars about fixed centers, past the pad; `grating()`'s ceil-based bar count can end the last bar past the pad). Validation at construction covers bar ordering only; steps that legitimately produce degenerate intermediates (`expandbars`, `apodizebars`) construct with `validate=False` internally, and the next validated step in the chain reports the problem.

Every module-level function taking `(starts, ends)` as its first arguments has a corresponding method, same name, same defaults. The functional API remains fully supported; `f(*g)` and `g.f()` are interchangeable.

## Generators

- `grating(period, dc, padx, ...)` — uniform grating with duty cycle, phase, pad gaps, optional apodization
- `kchirpgrating(p0, p1, ...)` — linearly chirped k-vector, exact per-period boundaries
- `apodizedgrating`, `gaussapodizedgrating` — amplitude apodization via duty cycle, using the d(dc) = sin(π·dc) relation
- `fixedbarapodized(Λ, gx, σ)` — Gaussian apodization with fixed half-period bars, poling-direction flips placed by delta-sigma matching of the integrated response
- `phasegrating`, `piphasegrating` — π phase-shifted (frequency-split) gratings; `spectralcomb` — double-slit-style comb
- `alternatinggrating`, `alternatinggrating2` — sectioned dual-period gratings
- `interleavedgrating`, `geometricinterleavedgrating` — dual-period sign(cosΛ₁ + cosΛ₂) gratings; the geometric version computes exact zero crossings from the carrier/envelope factorization
- `apodizedinterleavedgrating`, `discreteapodizedinterleavedgrating`, `geometricapodizedinterleavedgrating` — apodized dual-period gratings, by delta-sigma pole/no-pole decisions against cumulative amplitude targets or by exact boolean geometry
- `interleaveddeff(α)`, `interleavedd2fromd1(d1)` — the (d₁, d₂) efficiency tradeoff for interleaved gratings, exact via elliptic integrals
- `halfperiodgapgrating`, `phaseflipgrating`, `onoffs2grating`, `signedbars2grating`, `gratingfromxys`, `fixedresgrating2bars`

## Transforms

`mergetouchingbars`, `dropsmallbars` (lithographic tolerances), `expandbars`/`shrinkbars` (overpole compensation, constant or bar-size-dependent), `invertbarsgaps`, `breakupbars`/`breakupgaps` (maximum feature/gap size), `apodizebars` (triangle, trapezoidal, asin-triangle, asin-Gaussian duty-cycle profiles), `splitbarsalongperiodboundaries`, `reverse`, `binaryand`/`binaryor`.

## Analysis

- `ftgrating` — Fourier transform of the poling pattern: relative QPM amplitude or intensity vs period
- `gratingamplitude(g, Λ)` — local grating strength at period Λ, per-period average normalized to 1; the right tool for verifying apodization envelopes and interleaved-grating channel crosstalk
- `bestphase(g, Λ)` — phase offset maximizing overlap with cos(2πx/Λ)
- `grating2wave`, `gratingperiod2wave` — pattern and period-vs-position as `Wave`s
- `electrodegratingft` — response of an electrode-defined (gap-inverted) grating vs uniform reference

## Conventions

- Positions typically in µm; bars are poled (inverted) domains, gaps unpoled
- Bars are strictly ordered and non-overlapping; validation is on construction
- Relative amplitude of a bar with duty cycle dc is sin(π·dc); a full-length 50%-dc grating has amplitude 1 = 2/π of the unpatterned nonlinearity
- Default tolerances (1.0 µm) reflect what resolves lithographically while maintaining a resistive gap

## Tests

`gratingstests.py` contains the working examples above plus spectral studies (phase gratings vs length, alternating-grating channel spacing, apodization bandwidth comparisons, interleaved-grating cumulative response). Plotting requires `wavedata`; the phase-matching examples additionally require a `sellmeier` module for poling-period calculations. `gratings.py` run as a script executes `classtests()`, a fast geometry-only self-test with no plotting dependencies.
