// Headless PolyTrack 0.6.3 physics: patches the game's own simulation worker bundle to expose
// the raw createCarModel/updateCarModel step. Usage: const sim = await PtSim.load(capturePath)
const fs = require('fs'), path = require('path');
const VENDOR = path.join(__dirname, '..', 'vendor', 'pt063');

function decode(v) { // captured JSON -> typed arrays
  if (Array.isArray(v)) return v.map(decode);
  if (v && typeof v === 'object') {
    if (v.__t && v.b64) { const b = Buffer.from(v.b64, 'base64'); return new (globalThis[v.__t])(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); }
    const o = {}; for (const k in v) o[k] = decode(v[k]); return o;
  }
  return v;
}

function bootWorker() {
  let src = fs.readFileSync(path.join(VENDOR, 'simulation_worker.bundle.js'), 'utf8');
  const a = 'PolyTrackPhysics().then(', b = '$o.length=0,onmessage=r;';
  if (!src.includes(a) || !src.includes(b)) throw new Error('worker bundle layout changed');
  src = src.split('o=i(1903).Buffer,').join('o=globalThis.Buffer,');
  src = src.replace(a, 'PolyTrackPhysics({wasmBinary:globalThis.__wasm}).then(')
           .replace(b, b + 'globalThis.__pt={mod:t,create:s,step:n,handle:r,cars:e,Ko:Ko};');
  globalThis.self = globalThis; globalThis.JS_SHA256_NO_NODE_JS = true;
  globalThis.__wasm = fs.readFileSync(path.join(VENDOR, 'polytrack_physics.wasm'));
  globalThis.location = { href: 'file:///' + VENDOR.split(path.sep).join('/') + '/simulation_worker.bundle.js' };
  globalThis.postMessage = (m) => { globalThis.__out && globalThis.__out(m); };
  globalThis.addEventListener = () => {}; globalThis.onmessage = null;
  globalThis.importScripts = (f) => { (0, eval)(fs.readFileSync(path.join(VENDOR, f), 'utf8') + '\n;globalThis.PolyTrackPhysics=PolyTrackPhysics;'); };
  (0, eval)(src);
}

// CarState layout (see main.bundle.js decoder)
function parseState(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength); let i = 0;
  i = 4; const frames = u8[i] | u8[i+1] << 8 | u8[i+2] << 16; i += 3;
  const speedKmh = dv.getFloat32(i, true); i += 4;
  const fl = u8[i++]; const finished = !!(fl & 2);
  const wc = [!!(fl & 8), !!(fl & 16), !!(fl & 32), !!(fl & 64)];
  let finishFrames = null; if (finished) { finishFrames = u8[i] | u8[i+1] << 8 | u8[i+2] << 16; i += 3; }
  const nextCp = dv.getUint16(i, true); i += 2;
  const pos = [dv.getFloat32(i, true), dv.getFloat32(i+4, true), dv.getFloat32(i+8, true)]; i += 12;
  const quat = [dv.getFloat32(i, true), dv.getFloat32(i+4, true), dv.getFloat32(i+8, true), dv.getFloat32(i+12, true)]; i += 16;
  const nImp = u8[i++]; let maxImpulse = 0;
  for (let k = 0; k < nImp; k++) { maxImpulse = Math.max(maxImpulse, Math.abs(dv.getFloat32(i, true))); i += 4; }
  for (let k = 0; k < 4; k++) if (wc[k]) i += 24;
  const susp = [], suspVel = [], skid = [];
  for (let k = 0; k < 4; k++) { susp.push(dv.getFloat32(i, true)); i += 4; }
  for (let k = 0; k < 4; k++) { suspVel.push(dv.getFloat32(i, true)); i += 4; }
  i += 16; // wheelDeltaRotation
  for (let k = 0; k < 4; k++) { skid.push(dv.getFloat32(i, true)); i += 4; }
  const steering = dv.getFloat32(i, true);
  return { frames, speedKmh, finished, finishFrames, nextCp, pos, quat, maxImpulse, wc, susp, suspVel, skid, steering };
}

class PtSim {
  static async load(capturePath) {
    const cap = JSON.parse(fs.readFileSync(capturePath, 'utf8'));
    if (!globalThis.__pt) { bootWorker(); for (let k = 0; k < 200 && !globalThis.__pt; k++) await new Promise(r => setTimeout(r, 10)); }
    const s = new PtSim(); s.init = decode(cap.init); s.create = decode(cap.create);
    s.pt = globalThis.__pt; s.pt.handle({ data: s.init }); s.nextId = 1;
    s.track = s.pt.Ko.fromSaveString(s.create.trackData);
    return s;
  }
  reset() {
    if (this.id !== undefined) { this.pt.mod.ccall('deleteCarModel', 'void', ['number'], [this.id]); const i = this.pt.cars.findIndex(c => c.id === this.id); if (i >= 0) this.pt.cars.splice(i, 1); }
    this.id = this.nextId++;
    this.pt.handle({ data: { ...this.create, carId: this.id } });
    this.car = { id: this.id };
    return this.step(0, 1);
  }
  // keys: bitmask up=1,right=2,down=4,left=8,reset=16 ; advance n physics ticks (1 tick = 1 ms)
  step(keys, n = 1) {
    let st;
    for (let k = 0; k < n; k++) {
      const u8 = new Uint8Array(this.pt.step(this.car, { up: !!(keys & 1), right: !!(keys & 2), down: !!(keys & 4), left: !!(keys & 8), reset: !!(keys & 16) }));
      st = parseState(u8); if (st.finished) break;
    }
    return st;
  }
}
module.exports = { PtSim, parseState };
