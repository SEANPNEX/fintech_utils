# fintech_utils/momentum/config.py
from typing import Tuple
from pydantic import BaseModel, field_validator

class MomentumConfig(BaseModel):
    momentum_window: int = 126
    zscore_window: int = 63
    iv_window: int = 252
    dtm_range: Tuple[int, int] = (30, 45)        
    delta_range: Tuple[float, float] = (0.25, 0.40)
    es_target: float            
    var_alpha: float = 0.05
    es_alpha: float = 0.05
    iv_pctile_min: float = 0.5                  
    roll_min_dtm: int = 15
    top_n: int = 20

    @field_validator('delta_range')
    @classmethod
    def _check_delta(cls, v: Tuple[float, float]):
        lo, hi = v
        if not (0 <= lo < hi <= 1):
            raise ValueError('delta_range must be within [0,1] and lo < hi')
        return v

    @field_validator('dtm_range')
    @classmethod
    def _check_dtm(cls, v: Tuple[int, int]):
        lo, hi = v
        if not (lo > 0 and hi > lo):
            raise ValueError('dtm_range must be positive and lo < hi')
        return v