"""Synthetic adaptive-trajectory safety counterexample.

Proof-first bypasses review of a legitimately certified unlock action. That
changes which future requests the policy can make. NOT production security.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from statistics import NormalDist
import numpy as np
import argparse,json,math
from pathlib import Path

@dataclass(frozen=True)
class Config:
    episodes:int=200000
    seed:int=20261023
    false_denial_probability:float=.15
    unsafe_miss_probability:float=.08
    review_cost:float=1.
    proof_check_cost:float=.01
    transition_check_cost:float=.02

def wilson_interval(k:int,n:int,confidence:float=.95)->list[float]:
    if not(n>0 and 0<=k<=n and 0<confidence<1):
        raise ValueError('invalid binomial data')
    z=NormalDist().inv_cdf((1+confidence)/2);p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-radius),min(1.,center+radius)]

def experiment(config:Config)->dict:
    if config.episodes<=0:raise ValueError('invalid episode count')
    if not all(0<=v<=1 for v in (config.false_denial_probability,config.unsafe_miss_probability)):
        raise ValueError('invalid probabilities')
    if min(config.review_cost,config.proof_check_cost,config.transition_check_cost)<0:
        raise ValueError('invalid costs')
    n=config.episodes
    rng=np.random.default_rng(config.seed)
    wrongly_denied=rng.random(n)<config.false_denial_probability
    unsafe_missed=rng.random(n)<config.unsafe_miss_probability
    # Reviewer gates the authorized unlock; then may miss a later unsafe tool.
    unlocked_all=~wrongly_denied
    bad_all=unlocked_all&unsafe_missed
    # A SOUND certificate for unlock bypasses the first reviewer.
    bad_proof=unsafe_missed
    added=bad_proof&~bad_all
    assert np.array_equal(added,wrongly_denied&unsafe_missed)
    def row(name,bad,progress,reviews,shield=False):
        count=int(bad.sum())
        return dict(method=name,unsafe_episodes=count,
            unsafe_episode_rate=count/n,unsafe_episode_rate_ci95=wilson_interval(count,n),
            legitimate_progress_rate=float(progress.mean()),
            reviewer_calls_per_episode=float(reviews.mean()),
            cost_units_per_episode=float(config.review_cost*reviews.mean()+
                (config.proof_check_cost if name!='verify_all' else 0)+
                (config.transition_check_cost if shield else 0)))
    ones=np.ones(n,dtype=bool)
    rows=[row('verify_all',bad_all,unlocked_all,1+unlocked_all.astype(int)),
          row('sound_proof_first',bad_proof,ones,ones),
          row('sound_proof_plus_stateful_shield',np.zeros(n,dtype=bool),ones,ones,True)]
    k=int(added.sum())
    return dict(synthetic=True,config=asdict(config),results=rows,
       paired_added_unsafe_episodes=k,paired_risk_increase=k/n,
       paired_risk_increase_ci95=wilson_interval(k,n),
       exact_theoretical_risk_increase=config.false_denial_probability*config.unsafe_miss_probability,
       note='Stateful shield assumes a sound independent authorization engine, not a learned monitor.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=200000)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--out',type=Path,default=Path('results/v05/endogenous_counterexample.json'))
    args=p.parse_args()
    result=experiment(Config(episodes=args.episodes,seed=args.seed))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
