"""Finite-horizon, on-policy audit bookkeeping and transparent safety claims.

The model assumes each audit decision is secretly randomized *after* the policy
has proposed an action, with propensity p_t known to the auditor. The reviewer
may miss unsafe proposals with sensitivity >= q_min. An audit-positive result is
necessarily associated with a review; false positives make the bound looser,
not anti-conservative. The bound covers PAST realized proposal labels only.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import math

@dataclass
class HistoricalAuditor:
    delta:float=.05
    observations:int=0
    inverse_weighted_flags:float=0.
    squared_ranges:float=0.
    _history:list[dict]=field(default_factory=list,repr=False)

    def add(self, *, audit:bool, flag:bool, propensity:float, sensitivity_lower:float=1.0):
        if not (0 <self.delta<1): raise ValueError('delta must be in (0,1)')
        if not(0<propensity<=1 and 0<sensitivity_lower<=1): raise ValueError('invalid probabilities')
        if flag and not audit: raise ValueError('unaudited action cannot produce reviewer flag')
        self.observations+=1
        corrected=(float(flag)/(propensity*sensitivity_lower)) if audit else 0.
        self.inverse_weighted_flags+=corrected
        self.squared_ranges+=(1+1/(propensity*sensitivity_lower))**2
        self._history.append(dict(audit=audit,flag=flag,propensity=propensity,lower_sensitivity=sensitivity_lower))

    @property
    def upper_historical_rate(self)->float:
        n=self.observations
        if n==0: return 1.0
        # Union-bound over all positive integer times, alpha_n = alpha*6/(pi^2 n^2).
        alpha_n=self.delta*6/(math.pi**2*n*n)
        radius=math.sqrt(self.squared_ranges*math.log(1/alpha_n)/2)/n
        return min(1., self.inverse_weighted_flags/n+radius)

    @property
    def diagnostic(self)->dict:
        return dict(t=self.observations,upper_historical_rate=self.upper_historical_rate,
                    observed_flags=sum(x['flag'] for x in self._history),
                    number_audited=sum(x['audit'] for x in self._history),
                    bound_type='time-uniform upper bound on historical unauthorized proposal fraction; NOT future safety')


def coupled_execution(unsafe:bool,certified:bool,reviewer_rejects:bool)->tuple[bool,bool]:
    """Return unsafe execution under (verify-all, proof-first) on same action/reviewer.

    A 'certified' action bypasses the reviewer. The only possible extra safety
    error equals unsafe AND certified AND reviewer_rejects.
    """
    all_unsafe=unsafe and not reviewer_rejects
    proof_unsafe=unsafe and (certified or not reviewer_rejects)
    return bool(all_unsafe),bool(proof_unsafe)


def expected_extra_violations_upper_bound(*,number_unsafe_attempts:int,false_certificate_probability_upper:float)->float:
    """Union-free expected COUNT upper bound; assumptions apply per unsafe proposal.

    It need not assume reviewer independence: false certification on a bad action
    can add at most one additional unsafe execution.
    """
    if number_unsafe_attempts<0 or not (0<=false_certificate_probability_upper<=1):
        raise ValueError('invalid assumptions')
    return number_unsafe_attempts*false_certificate_probability_upper
