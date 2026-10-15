"""Road-aware progress field: graph over drivable road tiles, geodesic distance to each target."""
import json, os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVABLE = {"Road", "RoadWide", "RoadTurns", "Special"}
ROT = [lambda x, z: (x, z), lambda x, z: (z, -x), lambda x, z: (-x, -z), lambda x, z: (-z, x)]

class NavField:
    def __init__(self, parts, targets, part_info=None, reach_xz=1, reach_y=1):
        info = part_info or {int(k): v for k, v in json.load(open(os.path.join(ROOT, "data", "part_info.json"))).items()}
        tiles = set()
        for x, y, z, pid, rot, ax in parts:
            pi = info.get(pid)
            if pi is None or pi["cat"] not in DRIVABLE: continue
            for lo, hi in pi["tiles"]:
                for tx in range(lo[0], hi[0] + 1):
                    for ty in range(lo[1], hi[1] + 1):
                        for tz in range(lo[2], hi[2] + 1):
                            rx, rz = ROT[rot % 4](tx, tz); tiles.add((x + rx, y + ty, z + rz))
        self.tiles = np.array(sorted(tiles), dtype=np.float64); self.nodes = self.tiles * 5.0 + np.array([2.5, 0.0, 2.5])
        idx = {tuple(t): i for i, t in enumerate(self.tiles.astype(int))}
        rows, cols, w = [], [], []
        for (x, y, z), i in idx.items():
            for dx in range(-reach_xz, reach_xz + 1):
                for dz in range(-reach_xz, reach_xz + 1):
                    for dy in range(-reach_y, reach_y + 1):
                        j = idx.get((x + dx, y + dy, z + dz))
                        if j is not None and j != i:
                            rows.append(i); cols.append(j); w.append(5.0 * np.sqrt(dx * dx + dz * dz + dy * dy) + 1e-3)
        n = len(self.nodes); g = csr_matrix((w, (rows, cols)), shape=(n, n))
        tp = np.array([t["p"] for t in targets])
        self.tn = [self.nearest(p) for p in tp]  # target -> its road node
        self.field = [dijkstra(g, directed=True, indices=ti) for ti in self.tn]
        self.nbrs = [np.array([c for c in g.indices[g.indptr[i]:g.indptr[i + 1]]]) for i in range(n)]

    def nearest(self, pos):
        d = np.linalg.norm(self.nodes[:, [0, 2]] - np.array([pos[0], pos[2]]), axis=1) + 1.5 * np.abs(self.nodes[:, 1] + 2.0 - pos[1])
        return int(np.argmin(d))

    def dist(self, pos, ti):
        v = self.field[ti][self.nearest(pos)]; return float(v) if np.isfinite(v) else 1e4

    def lookahead(self, pos, ti, ahead=(40.0, 100.0)):
        f = self.field[ti]; cur = self.nearest(pos); walked = 0.0; out = []
        for a in ahead:
            while walked < a:
                nb = self.nbrs[cur]
                if len(nb) == 0: break
                j = nb[np.argmin(f[nb])]
                if f[j] >= f[cur] - 1e-6: break
                walked += float(np.linalg.norm(self.nodes[j] - self.nodes[cur])); cur = j
            out.append(self.nodes[cur].copy())
        return out
