"""Battery and fee parameters."""
from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_config(cls, cfg: Config) -> "Fees":
        f = cfg["fees"]
        return cls(toll_discharge=float(f["generation_toll_eur_per_mwh"]["value"]),
                   toll_charge=float(f["access_toll_charging_eur_per_mwh"]["value"]),
                   market=float(f["market_operator_fee_eur_per_mwh"]["value"]),
                   ivpee=float(f["ivpee_rate"]["value"]))
