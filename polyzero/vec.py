import numpy as np
from .env import PolyTrackEnv, OBS_DIM

class VecEnv:
    """N sim processes stepped concurrently (send all, then receive all)."""
    def __init__(self, n, **kw):
        self.envs = [PolyTrackEnv(**kw) for _ in range(n)]
        self.n, self.n_actions = n, self.envs[0].n_actions
    def reset(self):
        return np.stack([e.reset() for e in self.envs])
    def step(self, actions):
        for e, a in zip(self.envs, actions): e.step_send(a)
        obs, rew, done, infos = [], [], [], []
        for e in self.envs:
            o, r, d, info = e.step_recv()
            if d: info["terminal"] = True; o = e.reset()
            obs.append(o); rew.append(r); done.append(d); infos.append(info)
        return np.stack(obs), np.array(rew, np.float32), np.array(done, np.float32), infos
    def close(self):
        for e in self.envs: e.close()
