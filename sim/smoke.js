const { PtSim } = require('./ptsim');
(async () => {
  const sim = await PtSim.load(__dirname + '/../data/summer1_capture.json');
  console.log('determinism(builtin):', sim.pt.mod.ccall('testDeterminism', 'boolean'));
  const run = () => { let s = sim.reset(); const tr = [];
    for (let k = 0; k < 60; k++) { s = sim.step(1 | (k > 30 ? 2 : 0), 20); tr.push(s.pos.map(x => x.toFixed(4)).join(',')); } return [s, tr.join('|')]; };
  let t = Date.now(); const [s1, a] = run(); const dt = Date.now() - t;
  const [s2, b] = run();
  console.log('state after 1.2s:', JSON.stringify(s1)); console.log('ticks/s:', Math.round(1200 / (dt / 1000)), 'replay identical:', a === b);
  console.log('parts', sim.track.numberOfParts);
process.exit(0)})().catch(e => { console.error(e); process.exit(1) });
