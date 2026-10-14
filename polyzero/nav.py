"""Road-aware progress field: graph over placed part cells, geodesic distance to each target."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class NavField:
    def __init__(self, cells, targets, link_xz=45.0, link_y=60.0):
        self.cells = np.array(cells, dtype=np.float64)
        tp = np.array([t["p"] for t in targets])
        # snap targets as extra nodes
        self.nodes = np.vstack([self.cells, tp]); n = len(self.nodes)
        d_xz = np.linalg.norm(self.nodes[:, None, [0, 2]] - self.nodes[None, :, [0, 2]], axis=2)
        d_y = np.abs(self.nodes[:, None, 1] - self.nodes[None, :, 1])
        d3 = np.sqrt(d_xz ** 2 + d_y ** 2)
        adj = (d_xz <= link_xz) & (d_y <= link_y) & (d3 > 0)
        g = csr_matrix(np.where(adj, d3, 0.0))
        self.tidx = [len(self.cells) + i for i in range(len(tp))]
        self.field = [dijkstra(g, directed=False, indices=ti) for ti in self.tidx]  # [target][node] -> geodesic dist
        self.nbrs = [np.nonzero(adj[i])[0] for i in range(n)]

    def nearest(self, pos):
        d = np.linalg.norm(self.nodes[:, [0, 2]] - np.array([pos[0], pos[2]]), axis=1) + 0.3 * np.abs(self.nodes[:, 1] - pos[1])
        return int(np.argmin(d))

    def dist(self, pos, ti):
        i = self.nearest(pos); return float(self.field[ti][i]) if np.isfinite(self.field[ti][i]) else 1e4

    def lookahead(self, pos, ti, ahead=(40.0, 100.0)):
        """Walk downhill on the geodesic field; return points ~ahead metres along the route."""
        f = self.field[ti]; i = self.nearest(pos); out = []; walked = 0.0; cur = i
        for a in ahead:
            while walked < a:
                nb = self.nbrs[cur]
                if len(nb) == 0: break
                j = nb[np.argmin(f[nb])]
                if f[j] >= f[cur] - 1e-6: break
                walked += float(np.linalg.norm(self.nodes[j] - self.nodes[cur])); cur = j
            out.append(self.nodes[cur].copy())
        return out
