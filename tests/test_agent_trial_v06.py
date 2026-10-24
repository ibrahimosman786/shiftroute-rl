import unittest
from shiftroute_rl.agent_trial_v06 import Task,tasks,prompt,environment_step

class AgentTaskTests(unittest.TestCase):
    def test_three_stage_authorization(self):
        s=0
        for a in (0,1,2):
            s,r,bad=environment_step(s,a,False,False)
            self.assertFalse(bad)
            self.assertEqual(r,1.0)
        self.assertEqual(s,3)

    def test_weak_reward_mislabels_unauthorized(self):
        self.assertEqual(environment_step(1,3,True,True),(1,2.0,True))
        self.assertEqual(environment_step(1,3,False,True),(1,-3.0,True))

    def test_wrong_order_never_advances(self):
        self.assertEqual(environment_step(0,2,False,False),(0,-.3,False))

    def test_prompts_change_with_history(self):
        t=tasks(1,7,shifted=True)[0]
        self.assertNotEqual(prompt(t,0,'none'),prompt(t,1,'A completed'))
        self.assertEqual(tasks(10,8,True),tasks(10,8,True))

    def test_invalid(self):
        with self.assertRaises(ValueError):tasks(0,7,False)
        with self.assertRaises(ValueError):environment_step(3,3,False,False)
