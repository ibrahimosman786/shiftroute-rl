import unittest
from shiftroute_rl.trajectory_v05 import Config,experiment,wilson_interval

class TrajectoryCounterexampleTests(unittest.TestCase):
    def test_sound_proof_can_increase_episode_harm(self):
        x=experiment(Config(episodes=1000,false_denial_probability=1,unsafe_miss_probability=1))
        self.assertEqual([v['unsafe_episodes'] for v in x['results']],[0,1000,0])

    def test_no_false_denial_no_extra_harm(self):
        x=experiment(Config(episodes=1000,false_denial_probability=0,unsafe_miss_probability=.2))
        self.assertEqual(x['paired_added_unsafe_episodes'],0)

    def test_perfect_unsafe_reviewer(self):
        x=experiment(Config(episodes=1000,false_denial_probability=.4,unsafe_miss_probability=0))
        self.assertEqual([v['unsafe_episodes'] for v in x['results']],[0,0,0])

    def test_calibrated_simulation_matches_closed_form(self):
        x=experiment(Config(episodes=200000,seed=20261023))
        self.assertLess(abs(x['paired_risk_increase']-.012),.002)
        self.assertEqual(x['results'][2]['unsafe_episodes'],0)
        self.assertLess(x['results'][1]['reviewer_calls_per_episode'],
                        x['results'][0]['reviewer_calls_per_episode'])
        a,b=x['paired_risk_increase_ci95']
        self.assertTrue(a<=.012<=b)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):experiment(Config(episodes=0))
        with self.assertRaises(ValueError):experiment(Config(unsafe_miss_probability=1.01))
        with self.assertRaises(ValueError):wilson_interval(-1,100)
