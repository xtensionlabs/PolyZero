"""CEM/MPC planner over the true physics (snapshot/restore forking). Receding horizon, warm-started."""
import json, numpy as np
from .env import PolyTrackEnv, ACTIONS

KEYS = [t | s for t, s in ACTIONS]

class Planner:
    def __init__(self, envs, horizon=30, samples=48, iters=2, elite_frac=0.2, hold=None):
        self.envs, self.H, self.S, self.iters = envs, horizon, samples, iters
        self.ne = max(2, int(samples * elite_frac)); self.hold = hold or envs[0].tpa
        tp = envs[0].tp; self.tp = tp
        seg = [float(np.linalg.norm(tp[j] - tp[j - 1])) for j in range(1, len(tp))]
        self.rem = [sum(seg[j:]) for j in range(len(tp))]  # remaining straight-line path after reaching target j
        self.probs = np.full((horizon, len(KEYS)), 1.0 / len(KEYS))

    def potential(self, cp, d):
        j = min(cp, len(self.tp) - 1); return d + self.rem[j]

    def _eval(self, seqs):
        chunks = np.array_split(np.arange(len(seqs)), len(self.envs))
        for e, idx in zip(self.envs, chunks):
            e.proc.stdin.write(json.dumps({"cmd": "eval", "hold": self.hold, "seqs": [[KEYS[a] for a in seqs[i]] for i in idx]}) + "\n"); e.proc.stdin.flush()
        out = []
        for e in self.envs:
            r = json.loads(e.proc.stdout.readline())
            if "error" in r: raise RuntimeError(r["error"])
            out += r["res"]
        return np.array(out)

    def plan(self, prior=None, rng=np.random):
        e0 = self.envs[0]; st = e0.state
        phi0 = self.potential(st["nextCp"], e0._dist(st)[0])
        probs = self.probs.copy()
        if prior is not None: probs[0] = 0.5 * probs[0] + 0.5 * prior
        for _ in range(self.iters):
            cum = probs.cumsum(1); u = rng.random((self.S, self.H, 1))
            seqs = (u > cum[None]).sum(2).clip(0, len(KEYS) - 1)  # S x H
            res = self._eval(seqs)
            phi = np.array([self.potential(int(r[0]), r[1]) for r in res])
            score = (phi0 - phi) - 2.0 * res[:, 2] + 300.0 * res[:, 3] - 0.05 * res[:, 4] / 1000.0 * res[:, 3]
            el = seqs[np.argsort(-score)[:self.ne]]
            new = np.stack([np.bincount(el[:, t], minlength=len(KEYS)) for t in range(self.H)]) / self.ne
            probs = 0.3 * probs + 0.7 * new; probs = 0.97 * probs + 0.03 / len(KEYS)
        best = int(np.argmax(probs[0])); self.last_scores = score
        # receding horizon: shift, refill tail with uniform
        self.probs = np.vstack([probs[1:], np.full((1, len(KEYS)), 1.0 / len(KEYS))])
        return best, probs[0] / probs[0].sum()
