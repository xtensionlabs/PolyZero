"""Gym-style env over the headless PolyTrack physics (sim/server.js)."""
import json, subprocess, os, math
import numpy as np
from .nav import NavField

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# discrete action = (throttle/brake in {fwd, none, brake}) x (steer in {left, none, right})
ACTIONS = [(t, s) for t in (1, 0, 4) for s in (8, 0, 2)]  # key bits: up=1,right=2,down=4,left=8
OBS_DIM = 39

def qrot_inv(q, v):  # rotate world vector into car frame
    x, y, z, w = q; vx, vy, vz = v
    qx, qy, qz = -x, -y, -z
    tx = 2 * (qy * vz - qz * vy); ty = 2 * (qz * vx - qx * vz); tz = 2 * (qx * vy - qy * vx)
    return np.array([vx + w * tx + (qy * tz - qz * ty), vy + w * ty + (qz * tx - qx * tz), vz + w * tz + (qx * ty - qy * tx)])

def qrot_fwd(q):  # world-space forward (+Z local) of the car
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

class PolyTrackEnv:
    def __init__(self, capture="data/summer1_capture.json", ticks_per_action=40, max_seconds=180, stuck_seconds=12):
        self.proc = subprocess.Popen(["node", os.path.join(ROOT, "sim", "server.js")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, cwd=ROOT, bufsize=1)
        self.tpa, self.max_steps = ticks_per_action, int(max_seconds * 1000 / ticks_per_action)
        r = self._rpc({"cmd": "load", "capture": os.path.join(ROOT, capture)}); self.targets = r["targets"]
        self.nav = NavField(r["cells"], self.targets)
        self.tp = np.array([t["p"] for t in self.targets])
        self.n_actions = len(ACTIONS); self.stuck_steps = int(stuck_seconds * 1000 / ticks_per_action)

    def _rpc(self, q):
        self.proc.stdin.write(json.dumps(q) + "\n"); self.proc.stdin.flush()
        r = json.loads(self.proc.stdout.readline())
        if "error" in r: raise RuntimeError(r["error"])
        return r

    def _dist(self, st):
        i = min(st["nextCp"], len(self.tp) - 1)
        return float(np.linalg.norm(self.tp[i] - np.array(st["pos"]))), i

    def _obs(self, st):
        pos, q = np.array(st["pos"]), st["quat"]
        vel = (pos - self.prev_pos) / (self.tpa / 1000.0) if self.prev_pos is not None else np.zeros(3)
        self.prev_pos = pos
        v_loc = qrot_inv(q, vel) / 30.0
        o = [st["speedKmh"] / 100.0, *v_loc, st["steering"]]
        o += [float(c) for c in st["wc"]] + list(np.clip(np.array(st["susp"]) * 5, -2, 2)) + list(np.clip(st["skid"], 0, 1))
        o += [min(st["maxImpulse"] / 50.0, 2.0)]
        i = min(st["nextCp"], len(self.tp) - 1)
        for j in (i, min(i + 1, len(self.tp) - 1)):
            d = qrot_inv(q, self.tp[j] - pos); n = np.linalg.norm(d) + 1e-6
            o += [d[0] / n, d[1] / n, d[2] / n, min(n / 300.0, 2.0)]
        o += [1.0 if i == len(self.tp) - 1 else 0.0]
        up = qrot_inv(q, np.array([0.0, 1.0, 0.0])); o += [up[0], up[1], up[2]]
        fwd = qrot_fwd(q)
        o += [pos[0] / 320.0, pos[1] / 60.0, pos[2] / 320.0, fwd[0], fwd[1], fwd[2], vel[0] / 30.0, vel[1] / 30.0, vel[2] / 30.0]
        return np.array(o, dtype=np.float32)

    def reset(self):
        st = self._rpc({"cmd": "reset"})["state"]; self.prev_pos = None
        self.t = 0; self.prev_d, _ = self._dist(st); self.best_d, self.best_t = self.prev_d, 0; self.prev_cp = st["nextCp"]; self.state = st
        self.prev_pos = np.array(st["pos"])
        return self._obs(st)

    def step(self, a):
        self.step_send(a); return self.step_recv()

    def step_send(self, a):
        t, s = ACTIONS[int(a)]
        self.proc.stdin.write(json.dumps({"cmd": "step", "keys": t | s, "n": self.tpa}) + "\n"); self.proc.stdin.flush()

    def step_recv(self):
        r = json.loads(self.proc.stdout.readline())
        if "error" in r: raise RuntimeError(r["error"])
        return self._post(r["state"])

    def _post(self, st):
        self.t += 1; d, i = self._dist(st)
        r = 0.0
        if st["nextCp"] > self.prev_cp: r += 50.0 * (st["nextCp"] - self.prev_cp); self.prev_d = d  # reset shaping on target switch
        r += 1.0 * (self.prev_d - d) / 10.0          # progress toward target (per 10 m)
        r -= 0.02                                     # time pressure
        r -= 0.5 * min(st["maxImpulse"] / 50.0, 2.0)  # collisions
        up_y = 1 - 2 * (st["quat"][0] ** 2 + st["quat"][2] ** 2)
        flipped = up_y < 0.2
        if flipped: r -= 5.0
        if d < self.best_d - 1.0 or st["nextCp"] > self.prev_cp: self.best_d, self.best_t = d, self.t
        if st["nextCp"] > self.prev_cp: self.best_d = d
        stuck = self.t - self.best_t > self.stuck_steps
        if stuck: r -= 5.0
        done = bool(st["finished"] or flipped or stuck or self.t >= self.max_steps)
        if st["finished"]: r += 100.0
        self.prev_d, self.prev_cp, self.state = d, st["nextCp"], st
        return self._obs(st), r, done, {"finished": st["finished"], "cp": st["nextCp"], "frames": st["frames"]}

    def close(self):
        try: self.proc.stdin.close(); self.proc.kill()
        except Exception: pass
