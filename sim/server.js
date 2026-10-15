// JSON-lines server around PtSim. Requests: {cmd:'load',capture}, {cmd:'reset'}, {cmd:'step',keys,n}
const readline = require('readline');
const { PtSim } = require('./ptsim');
let sim = null, TP = [];
function targets(sim) {
  const det = {}; sim.init.trackParts.forEach(p => { if (p.detector) det[p.id] = p.detector; });
  const cps = [], fin = [];
  sim.track.forEachPart((x, y, z, id, rot, ax, col, co) => {
    const d = det[id]; if (!d) return;
    const c = d.center, r = (rot | 0) % 4; // quarter turns about Y (convention validated empirically)
    const [cx, cz] = [[c[0], c[2]], [c[2], -c[0]], [-c[0], -c[2]], [-c[2], c[0]]][r];
    const p = [x * 5 + cx, y * 5 + c[1], z * 5 + cz];
    (d.type === 0 ? cps : fin).push({ co, p, rot: r, w: d.size[0] });
  });
  cps.sort((a, b) => a.co - b.co);
  return cps.concat(fin);
}
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', async (line) => {
  let q; try { q = JSON.parse(line); } catch (e) { return; }
  try {
    let out;
    if (q.cmd === 'load') { sim = await PtSim.load(q.capture); const cells = []; sim.track.forEachPart((x, y, z, id, rot, ax) => cells.push([x, y, z, id, rot | 0, ax | 0])); TP = targets(sim); out = { ok: true, targets: TP, cells }; }
    else if (q.cmd === 'reset') out = { state: sim.reset() };
    else if (q.cmd === 'step') out = { state: sim.step(q.keys | 0, q.n || 1) };
    else if (q.cmd === 'eval') { // roll out candidate key sequences from the current state; state is restored afterwards
      const H = sim.pt.mod.HEAPU8, snap = H.slice(), res = [];
      for (const seq of q.seqs) {
        H.set(snap); let st, imp = 0;
        for (const keys of seq) { st = sim.step(keys, q.hold); imp += Math.min(st.maxImpulse / 50, 2); if (st.finished) break; }
        const i = Math.min(st.nextCp, TP.length - 1), p = TP[i].p;
        const d = Math.hypot(p[0] - st.pos[0], p[1] - st.pos[1], p[2] - st.pos[2]);
        res.push([st.nextCp, d, imp, st.finished ? 1 : 0, st.frames, st.speedKmh]);
      }
      H.set(snap); out = { res };
    }
    process.stdout.write(JSON.stringify(out) + '\n');
  } catch (e) { process.stdout.write(JSON.stringify({ error: String(e.stack) }) + '\n'); }
});
