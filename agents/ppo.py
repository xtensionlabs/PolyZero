"""Compact PPO (discrete actions) on the headless PolyTrack env. CPU-friendly."""
import sys, os, time, csv, argparse; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from polyzero.vec import VecEnv
from polyzero.env import OBS_DIM

class Net(nn.Module):
    def __init__(self, n_act, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi, self.v = nn.Linear(h, n_act), nn.Linear(h, 1)
    def forward(self, x):
        z = self.body(x); return self.pi(z), self.v(z).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=10); ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--name", default="run0")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--capture", default="data/summer1_capture.json")
    a = ap.parse_args(); torch.set_num_threads(2); torch.manual_seed(0)
    env = VecEnv(a.envs, capture=a.capture); net = Net(env.n_actions)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    os.makedirs(f"runs/{a.name}", exist_ok=True); log = csv.writer(open(f"runs/{a.name}/log.csv", "w", newline=""))
    log.writerow(["iter", "sec", "env_steps", "mean_ret", "max_cp", "finishes", "best_finish_s", "ent"])
    obs = torch.tensor(env.reset()); ep_ret = np.zeros(a.envs); rets, best_fin, t0 = [], 1e9, time.time()
    gamma, lam, T, N = 0.995, 0.95, a.steps, a.envs
    for it in range(a.iters):
        O = torch.zeros(T, N, OBS_DIM); A = torch.zeros(T, N, dtype=torch.long); LP = torch.zeros(T, N)
        R = torch.zeros(T, N); D = torch.zeros(T, N); V = torch.zeros(T + 1, N); maxcp = 0; fins = 0
        for t in range(T):
            with torch.no_grad():
                lg, v = net(obs); dist = torch.distributions.Categorical(logits=lg); act = dist.sample()
            o2, r, d, infos = env.step(act.numpy())
            O[t], A[t], LP[t], V[t], R[t], D[t] = obs, act, dist.log_prob(act), v, torch.tensor(r), torch.tensor(d)
            ep_ret += r
            for i, inf in enumerate(infos):
                maxcp = max(maxcp, inf["cp"])
                if d[i]:
                    rets.append(ep_ret[i]); ep_ret[i] = 0
                    if inf["finished"]: fins += 1; best_fin = min(best_fin, inf["frames"] / 1000)
            obs = torch.tensor(o2)
        with torch.no_grad(): V[T] = net(obs)[1]
        adv = torch.zeros(T, N); g = 0
        for t in reversed(range(T)):
            delta = R[t] + gamma * V[t + 1] * (1 - D[t]) - V[t]; g = delta + gamma * lam * (1 - D[t]) * g; adv[t] = g
        ret = adv + V[:T]
        fo, fa, fl, fadv, fret = O.reshape(-1, OBS_DIM), A.reshape(-1), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        for _ in range(4):
            perm = torch.randperm(T * N)
            for mb in perm.split(512):
                lg, v = net(fo[mb]); dist = torch.distributions.Categorical(logits=lg)
                ratio = (dist.log_prob(fa[mb]) - fl[mb]).exp()
                pl = -torch.min(ratio * fadv[mb], ratio.clamp(0.8, 1.2) * fadv[mb]).mean()
                loss = pl + 0.5 * (v - fret[mb]).pow(2).mean() - a.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        mr = float(np.mean(rets[-20:])) if rets else 0.0
        log.writerow([it, round(time.time() - t0), (it + 1) * T * N, round(mr, 2), maxcp, fins, round(best_fin, 2) if best_fin < 1e8 else "", round(float(dist.entropy().mean()), 3)])
        sys.stdout.flush(); print(f"it {it} t={time.time()-t0:.0f}s ret={mr:.1f} maxcp={maxcp} fin={fins} best={best_fin:.1f}", flush=True)
        if it % 10 == 0: torch.save(net.state_dict(), f"runs/{a.name}/net.pt")
    torch.save(net.state_dict(), f"runs/{a.name}/net.pt"); env.close()
if __name__ == "__main__": main()
