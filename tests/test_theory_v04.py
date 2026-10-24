import math, unittest
import numpy as np
from shiftroute_rl.theory_v04 import HistoricalAuditor,coupled_execution,expected_extra_violations_upper_bound

class TestFormalSafetyBoundary(unittest.TestCase):
    def test_all_binary_cases_sound_equivalence(self):
        for bad in (False,True):
            for rejected in (False,True):
                for certified in (False,True):
                    if bad and certified: continue  # independent sound certification
                    a,b=coupled_execution(bad,certified,rejected)
                    self.assertEqual(a,b)
    def test_unsound_certification_can_increase_bad_executions(self):
        self.assertEqual(coupled_execution(True,True,True),(False,True))
    def test_bypass_probability_expected_harm_bound(self):
        self.assertEqual(expected_extra_violations_upper_bound(number_unsafe_attempts=100,false_certificate_probability_upper=.01),1)
    def test_audit_history_bound_covers_adaptive_sequences(self):
        # Predictable p_t but adaptive unauthorized proposals; audit coins
        # sampled independently AFTER the proposal. q=0.9 detecting misses.
        # Tiny Monte Carlo is a *test*, not evidence of rigorous coverage.
        n=300; count_missed=0
        for seed in range(100):
            rng=np.random.default_rng(seed)
            checker=HistoricalAuditor(delta=.05)
            bad_prev=False
            unsafe_total=0
            for t in range(n):
                bad=bool(rng.random() < (.35 if bad_prev else .08))
                bad_prev=bad
                unsafe_total+=bad
                propensity=.2 if t<150 else .5
                audit=bool(rng.random()<propensity)
                flag=bool(audit and bad and rng.random()<.9)
                checker.add(audit=audit,flag=flag,propensity=propensity,sensitivity_lower=.85)
            count_missed+=checker.upper_historical_rate < unsafe_total/n
        self.assertLessEqual(count_missed,1)
    def test_conservative_empty(self):
        self.assertEqual(HistoricalAuditor().upper_historical_rate,1)
    def test_invalid_report(self):
        d=HistoricalAuditor()
        with self.assertRaises(ValueError):d.add(audit=False,flag=True,propensity=.5)
    def test_imperfect_sensitivity_is_not_a_production_certificate(self):
        d=HistoricalAuditor()
        for t in range(30): d.add(audit=True,flag=False,propensity=1.,sensitivity_lower=.8)
        self.assertGreater(d.upper_historical_rate,0)
        self.assertIn('NOT future',d.diagnostic['bound_type'])
