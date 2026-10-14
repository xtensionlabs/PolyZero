import sys, os; sys.path.insert(0, '.')
import numpy as np, torch
from polyzero.env import PolyTrackEnv
from agents.ppo import Net
name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
env = PolyTrackEnv(max_seconds=120); net = Net(env.n_actions); net.load_state_dict(torch.load(f'runs/{name}/net.pt')); net.eval()
for ep in range(n):
    o = env.reset(); d = False; path = []
    while not d:
        with torch.no_grad(): lg, _ = net(torch.tensor(o)[None])
        a = torch.distributions.Categorical(logits=lg).sample().item()
        o, r, d, info = env.step(a)
        if env.t % 100 == 0: path.append([round(x) for x in env.state['pos']])
    print(f"ep{ep} cp={info['cp']} fin={info['finished']} t={info['frames']/1000:.1f}s end={[round(x) for x in env.state['pos']]} v={env.state['speedKmh']:.0f}", 'path', path[:14])
env.close()
