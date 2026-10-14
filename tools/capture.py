"""Capture worker Init/CreateCar messages from the real game (v0.6.3) into data/capture.json.
Usage: python tools/capture.py [--list]   (--list just prints visible menu text)"""
import sys, json, base64, time
from playwright.sync_api import sync_playwright

HOOK = r"""
(() => {
  window.__cap = [];
  const enc = (v, d=0) => {
    if (ArrayBuffer.isView(v)) return {__t: v.constructor.name, b64: (()=>{const u=new Uint8Array(v.buffer,v.byteOffset,v.byteLength);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s)})()};
    if (Array.isArray(v)) return v.map(x => enc(x, d+1));
    if (v && typeof v === 'object') { const o = {}; for (const k in v) o[k] = enc(v[k], d+1); return o; }
    return v;
  };
  const P = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function(m, ...r) {
    try { if (m && m.messageType !== undefined) window.__cap.push({ts: performance.now(), m: enc(m)}); } catch (e) { window.__err = String(e); }
    return P.call(this, m, ...r);
  };
})();
"""

def jsclick(pg, text):
    ok = pg.evaluate("t => { const e=[...document.querySelectorAll('p,span,div,button')].find(x=>x.children.length==0&&x.innerText.trim()===t); if(!e) return false; (e.closest('button')||e).click(); return true }", text)
    print('click', text, ok)

def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.add_init_script(HOOK)
        pg.goto("https://app-polytrack.kodub.com/0.6.3/", wait_until="networkidle")
        time.sleep(3)
        if "--list" in sys.argv:
            print(pg.evaluate("() => [...document.querySelectorAll('button,[role=button],.button,a')].map(e=>e.innerText.trim()).filter(Boolean)"))
            print(pg.evaluate("() => document.body.innerText.slice(0,400)"))
        else:
            jsclick(pg, "Play"); time.sleep(4)
            cap = pg.evaluate("() => window.__cap")
            out = {"init": next(c["m"] for c in cap if c["m"]["messageType"] == 0),
                   "create": next(c["m"] for c in cap if c["m"]["messageType"] == 3)}
            json.dump(out, open("data/summer1_capture.json", "w"))
            print({k: {kk: (type(vv).__name__ if not isinstance(vv, list) else len(vv)) for kk, vv in v.items()} for k, v in out.items()})
        b.close()
main()
