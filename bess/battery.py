"""Battery and fee parameters."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Battery:
    power: float
    energy: float
    eta_c: float
    eta_d: float
    soc_min: float
    soc_max: float
    soc_target: float
    degradation: float
    max_cycles: float
    reserve_h: float

    @classmethod
    def from_config(cls, cfg: Config) -> "Battery":
        b = cfg["battery"]
        e = float(b["energy_mwh"])
        return cls(power=float(b["power_mw"]), energy=e, eta_c=float(b["eta_charge"]), eta_d=float(b["eta_discharge"]),
                   soc_min=e * float(b["soc_min_frac"]), soc_max=e * float(b["soc_max_frac"]),
                   soc_target=e * float(b["soc_initial_frac"]), degradation=float(b["degradation_eur_per_mwh"]),
                   max_cycles=float(b["max_cycles_per_day"]), reserve_h=float(b["afrr_full_activation_h"]))


@dataclass(frozen=True)
class Fees:
    toll_discharge: float
    toll_charge: float
    market: float
    ivpee: float
    ivpee_schedule: tuple = ()

    @classmethod
    def from_config(cls, cfg: Config) -> "Fees":
        f = cfg["fees"]
        sched = (f["ivpee_rate"].get("schedule") or {}).items()
        return cls(toll_discharge=float(f["generation_toll_eur_per_mwh"]["value"]),
                   toll_charge=float(f["access_toll_charging_eur_per_mwh"]["value"]),
                   market=float(f["market_operator_fee_eur_per_mwh"]["value"]),
                   ivpee=float(f["ivpee_rate"]["value"]),
                   ivpee_schedule=tuple(sorted((date.fromisoformat(str(d)), float(r)) for d, r in sched)))

    def on(self, day: date) -> "Fees":
        """Fees with the IVPEE rate in force on the day."""
        rate = self.ivpee
        for d, r in self.ivpee_schedule:
            if day >= d:
                rate = r
        return replace(self, ivpee=rate)


def ivpee_base(energy_revenue: np.ndarray, cap_up, cap_dn, r_up, r_dn, a_up, a_dn, e_up, e_dn, dt: float) -> float:
    """Taxable income: net energy revenue of each quarter-hour plus aFRR capacity and activation income."""
    afrr = (np.maximum(cap_up, 0) * r_up + np.maximum(cap_dn, 0) * r_dn
            + np.maximum(e_up, 0) * a_up + np.maximum(-e_dn, 0) * a_dn)
    return float((np.maximum(energy_revenue, 0).sum() + np.nansum(afrr)) * dt)
