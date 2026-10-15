"""Drive a lap with the CEM planner; log (obs, action, planner-probs) for distillation."""
import sys, os, time, argparse, pickle; sys.path.insert(0, '.')
import numpy as np
from polyzero.env import PolyTrackEnv
from polyzero.planner import Planner
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=10); ap.add_argument('--tpa', type=int, default=50)
ap.add_argument('--horizon', type=int, default=30); ap.add_argument('--samples', type=int, default=60)
ap.add_argument('--iters', type=int, default=2); ap.add_argument('--max_seconds', type=int, default=120)
ap.add_argument('--out', default='data/plan_lap.pkl'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
envs = [PolyTrackEnv(ticks_per_action=a.tpa, max_seconds=a.max_seconds) for _ in range(a.workers)]
pl = Planner(envs, horizon=a.horizon, samples=a.samples, iters=a.iters)
obs = [e.reset() for e in envs]; log = []; t0 = time.time(); done = False; step = 0; ret = 0
while not done:
    act, pr = pl.plan(rng=rng)
    log.append((obs[0].copy(), act, pr))
    for e in envs: e.step_send(act)
    outs = [e.step_recv() for e in envs]; obs = [o[0] for o in outs]; r, done, info = outs[0][1], outs[0][2], outs[0][3]; ret += r; step += 1
    if step % 20 == 0:
        s = envs[0].state; print(f"step {step} sim_t={s['frames']/1000:.1f}s wall={time.time()-t0:.0f}s cp={info['cp']} pos={[round(x) for x in s['pos']]} v={s['speedKmh']:.0f} ret={ret:.1f}", flush=True)
print('DONE', info, 'steps', step, 'ret', round(ret, 1), 'wall', round(time.time() - t0))
pickle.dump({'log': log, 'info': info, 'tpa': a.tpa}, open(a.out, 'wb'))
for e in envs: e.close()
