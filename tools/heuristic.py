"""Sanity-check env + target geometry with a trivial steer-to-target driver."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from polyzero.env import PolyTrackEnv, ACTIONS
env = PolyTrackEnv(max_seconds=60); print('targets', [(t['co'], [round(x) for x in t['p']]) for t in env.targets])
for sign in (1, -1):
    o = env.reset(); tot = 0; t0 = time.time()
    for k in range(env.max_steps):
        lat = o[18]  # next-target direction x in car frame (see _obs layout)
        steer = 0 if abs(lat) < 0.08 else (1 if lat * sign > 0 else -1)
        a = ACTIONS.index((1, {1: 2, 0: 0, -1: 8}[steer]))
        o, r, d, info = env.step(a); tot += r
        if d: break
    print('sign', sign, 'steps', k, 'return', round(tot, 1), info, 'sps', round(k / (time.time() - t0)))
env.close()
