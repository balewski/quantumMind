# Guppy Zoo Workflow

This directory contains the three-stage workflow for Guppy zoo benchmark jobs.

## Files

- `submit_qtuum_zoo.py`
  Builds GHZ or QFT benchmark programs and either:
  runs them locally with the Selene `ideal` or `noisy` simulator, or submits them to Quantinuum Nexus on `Helios-1E`.
  It writes submission metadata to `out/jobs/*.qtuum.npz`.

- `retrieve_qtuum_zoo.py`
  Loads submission metadata from `out/jobs/*.qtuum.npz`, queries the Nexus job, downloads raw counts, records job QA metadata, and writes packed measurement counts to `out/meas/*.meas.npz`.

- `postproc_zoo.py`
  Loads packed counts from `out/meas/*.meas.npz`, performs task-specific postprocessing for `ghz` or `qft`, stores the result in `expD['prob']`, and writes `out/post/*.post.npz`.

## Typical Flow

1. Submit or run a benchmark:

```bash
./submit_qtuum_zoo.py -t ghz -q 3 -n 100 -b ideal
./submit_qtuum_zoo.py -t qft 3 -q 12 -n 100 -b noisy
./submit_qtuum_zoo.py -t qft 3 -q 12 -n 100 -b Helios-1E -E
```

2. Retrieve raw Nexus results:

```bash
./retrieve_qtuum_zoo.py --expName <expName>
```

3. Postprocess the saved measurement counts:

```bash
./postproc_zoo.py --expName <expName>
```

## Notes

- Local `noisy` simulation uses the Selene `DepolarizingErrorModel`.
- `retrieve_qtuum_zoo.py` does not do benchmark-specific analysis.
- GHZ and QFT analysis lives in `postproc_zoo.py`.
