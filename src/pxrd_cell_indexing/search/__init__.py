"""B1: independent, peak-driven global candidate search (v3 §11 / v4 §6.1).

Deliberately kept out of ``model/`` and out of the training ``forward`` path:
this is a classical-indexing-style search over observed peak positions only,
not a learned component. See ``qsearch.py`` for the core algorithm and
``scripts/run_b1_s0_synthetic.py`` for the synthetic B1-S0 Gate test.
"""
