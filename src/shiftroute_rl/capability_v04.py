"""Fail-closed, context-bound HMAC capabilities for simulated agent tool calls.

The issuer is a trusted policy decision point, NEVER the LLM. The verifier
only receives an authenticated token and the actual request. Tokens are
single-use, scoped, time-bounded, policy-versioned and argument-bound.

HMAC is symmetric: both services must protect the same secret and the verifier
is also cryptographically capable of issuing tokens. Use an asymmetric or
hardware-backed construction if the verifier lies outside the trust boundary.
Do not use this illustrative implementation to protect live systems.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, hmac, json, secrets, time
from typing import Any


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class Action:
    session: str
    user: str
    tool: str
    resource: str
    arguments: dict[str, Any]

    def digest(self) -> str:
        return hashlib.sha256(canonical(self.__dict__)).hexdigest()


class Policy:
    """Example trusted access-control rules, independent from LLM confidence."""
    def __init__(self, *, version: str = 'v1', permissions: dict[str, set[tuple[str, str]]] | None = None):
        self.version=version
        self.permissions=permissions or {}

    def permits(self, action: Action) -> bool:
        return (action.tool, action.resource) in self.permissions.get(action.user, set())


class TrustedIssuer:
    def __init__(self, key: bytes, policy: Policy, *, audience: str = 'shiftroute-gate', ttl: int = 30):
        if len(key)<32: raise ValueError('HMAC key must have >= 256 bits')
        if ttl<=0: raise ValueError('TTL must be positive')
        self._key,self.policy,self.audience,self.ttl=key,policy,audience,ttl

    def issue(self, action: Action, *, now: int | None=None) -> str | None:
        if not self.policy.permits(action):
            return None
        now=int(time.time() if now is None else now)
        payload=dict(schema='sr-cap-v1',aud=self.audience,policy=self.policy.version,
                     action_sha256=action.digest(),issued_at=now,expires_at=now+self.ttl,
                     nonce=secrets.token_hex(16))
        signature=hmac.new(self._key,canonical(payload),hashlib.sha256).hexdigest()
        return json.dumps(dict(payload=payload,signature=signature),sort_keys=True)


class CapabilityGate:
    def __init__(self,key: bytes, *, policy_version: str='v1',audience: str='shiftroute-gate',max_clock_skew: int=0):
        if len(key)<32: raise ValueError('HMAC key must have >= 256 bits')
        self._key,self.policy_version,self.audience,self.max_clock_skew=key,policy_version,audience,max_clock_skew
        self._used:set[str]=set()

    def authorize(self,token: str | None,action: Action,*,now:int|None=None) -> bool:
        # Fail closed on malformed, truncated, forged, stale and previously used tokens.
        if token is None or len(token)>4096: return False
        now=int(time.time() if now is None else now)
        try:
            record=json.loads(token)
            if set(record)!={'payload','signature'}: return False
            p,s=record['payload'],record['signature']
            if not isinstance(p,dict) or not isinstance(s,str): return False
            if set(p)!={'schema','aud','policy','action_sha256','issued_at','expires_at','nonce'}: return False
            expected=hmac.new(self._key,canonical(p),hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected,s): return False
            if p['schema']!='sr-cap-v1' or p['aud']!=self.audience or p['policy']!=self.policy_version: return False
            if p['action_sha256']!=action.digest(): return False
            if not isinstance(p['nonce'],str) or len(p['nonce'])!=32: return False
            if not isinstance(p['issued_at'],int) or not isinstance(p['expires_at'],int): return False
            if p['issued_at']>now+self.max_clock_skew or p['expires_at']<=now: return False
            if p['expires_at']<=p['issued_at'] or p['nonce'] in self._used: return False
            self._used.add(p['nonce'])
            return True
        except (TypeError,KeyError,ValueError,OverflowError):
            return False
