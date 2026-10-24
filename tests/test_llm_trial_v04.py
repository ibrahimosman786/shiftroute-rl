import unittest
import numpy as np
from shiftroute_rl.llm_trial_v04 import examples,scores_from_labels,evaluate
class TestPretrainedTrialUtilities(unittest.TestCase):
    def test_splits_have_new_prompts(self):
        a=examples(100,7,shifted=False)
        b=examples(100,7,shifted=True)
        self.assertFalse(set(x.prompt for x in a)&set(x.prompt for x in b))
    def test_verified_feedback_matches_label(self):
        xs=examples(50,43)
        correct=np.array([0 if x.authorized else 1 for x in xs])
        self.assertTrue(np.all(scores_from_labels(correct,xs,'verified')==1.))
        self.assertEqual(evaluate(correct,xs)['unauthorized_approval_rate'],0)
    def test_proxy_reward_can_incentivize_unsafe_approval(self):
        xs=examples(400,100,shifted=True)
        unauthorized_without_alarm=[x for x in xs if not x.authorized and not x.monitor_detects]
        self.assertGreater(len(unauthorized_without_alarm),30)
        self.assertTrue(np.all(scores_from_labels(np.zeros(len(unauthorized_without_alarm)),unauthorized_without_alarm,'weak')==1))
    def test_invalid_feedback(self):
        with self.assertRaises(ValueError):scores_from_labels(np.array([0]),examples(2,1),'verified')
