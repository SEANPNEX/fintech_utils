# FinTech Utils

A Python package providing utilities for financial data analysis, including momentum strategies and option pricing models.

## Features
- Momentum calculation functions
- Option pricing models (Binomial Tree, Black-Scholes-Merton)

## Installation
```bash
pip install git+https://github.com/SEANPNEX/fintech_utils.git
```

## Usage
```python
from fintech_utils.momentum import compute_momentum, compute_vol_adj_momentum
from fintech_utils.findata.options import binomial_tree_option_price, bsm
```

## Momentum Module
- `compute_momentum(prices)`: Calculate 12-1 momentum and various volatility metrics
- `compute_vol_adj_momentum(mom_12_1, vol_realized)`: Calculate volatility-adjusted momentum


## Options Module
- `binomial_tree_option_price(...)`: Price options using the Binomial Tree model
- `bsm(S, X, T, r, sigma)`: Price options using the Black-Scholes-Merton model