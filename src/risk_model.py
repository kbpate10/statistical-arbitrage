"""Position sizing and risk limits.

Sizing logic
------------
I risk 2% of CURRENT equity per trade, where "risk" means the expected
loss if the stop is hit. I size off the stop distance rather than off
notional because it normalizes trades across pairs: a volatile spread
gets a smaller position than a quiet one, and each trade contributes
comparably to portfolio risk. The stop sits (STOP_Z - ENTRY_Z) z-units
away, and one z-unit of spread movement is worth `spread_std` dollars per
unit of spread, so:

    stop_distance_$  = (STOP_Z - ENTRY_Z) * spread_std * units
    units            = (RISK_PER_TRADE * equity) / ((STOP_Z - ENTRY_Z) * spread_std)

where one "unit" of the spread means long 1 share of y and short beta
shares of x (or the reverse). I cap gross dollar exposure per pair at
equity / MAX_CONCURRENT_POSITIONS so three concurrent pairs can never
exceed 1x gross leverage in total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config


def position_units(equity: float, spread_std: float, beta: float,
                   price_y: float, price_x: float) -> float:
    """Number of spread units to trade. 0 if the trade can't be sized."""
    if spread_std <= 0 or not pd.notna(spread_std):
        return 0.0
    stop_distance = (config.STOP_Z - config.ENTRY_Z) * spread_std
    units = (config.RISK_PER_TRADE * equity) / stop_distance

    # Cap gross exposure (|y leg| + |x leg|) per pair.
    gross_per_unit = price_y + abs(beta) * price_x
    max_units = (equity / config.MAX_CONCURRENT_POSITIONS) / gross_per_unit
    return min(units, max_units)


@dataclass
class Position:
    pair: str            # "Y/X"
    y: str
    x: str
    direction: int       # +1 long spread, -1 short spread
    units: float
    beta: float          # hedge ratio FROZEN at entry
    entry_date: pd.Timestamp
    entry_z: float
    entry_price_y: float
    entry_price_x: float
    entry_cost: float    # transaction costs paid at entry ($)
    features: dict | None = None  # decision-time features (meta-labeling)

    def market_value(self, price_y: float, price_x: float) -> float:
        """P&L since entry, before exit costs.

        Long spread: +units*y - units*beta*x  =>  pnl = units * (dy - beta*dx)
        """
        dy = price_y - self.entry_price_y
        dx = price_x - self.entry_price_x
        return self.direction * self.units * (dy - self.beta * dx)

    def gross_notional(self, price_y: float, price_x: float) -> float:
        return self.units * (price_y + abs(self.beta) * price_x)


def transaction_cost(gross_notional: float) -> float:
    """Cost + slippage for ONE side (entry or exit) of a pair trade, in $."""
    bps = config.COST_BPS_PER_SIDE + config.SLIPPAGE_BPS
    return gross_notional * bps / 10_000.0


@dataclass
class PortfolioState:
    """Open-position bookkeeping with the max-concurrency limit."""
    equity: float = config.INITIAL_CAPITAL
    positions: dict[str, Position] = field(default_factory=dict)

    def can_open(self, pair: str) -> bool:
        return (pair not in self.positions
                and len(self.positions) < config.MAX_CONCURRENT_POSITIONS)
