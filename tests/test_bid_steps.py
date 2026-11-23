"""Bid curves with a limit on the number of lines per period."""
import numpy as np
import pandas as pd

from bess.bids import curve


def test_curve_keeps_step_limit_and_total_quantity():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-08-08", periods=4, freq="15min", tz="UTC")
    P = rng.uniform(50, 200, (30, 4))
    Q = np.argsort(np.argsort(P, axis=0), axis=0) / 29.0 * 10.0 - 5.0
    lim = {"min": -500.0, "max": 4000.0}
    full = curve("da", idx, P, Q, lim)
    cut = curve("ida2", idx, P, Q, lim, max_steps=10)
    assert full.groupby("delivery_start").size().max() > 10
    assert cut.groupby("delivery_start").size().max() <= 10
    net = lambda b: (b["quantity_mw"] * b["side"].map({"sell": 1.0, "buy": -1.0})).groupby(b["delivery_start"]).sum()
    assert np.allclose(net(full), net(cut))
