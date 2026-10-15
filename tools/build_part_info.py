"""Parse part id -> (category, tile footprint) from the game bundle into data/part_info.json."""
import re, json
s = open('vendor/pt063/main.bundle.js', encoding='utf8').read()
m = re.search(r'e\[e\.Straight=0\]="Straight"', s); a = s.rfind('function', 0, m.start()); seg = s[m.start() - 5: m.start() + 12000]
enum = {n: int(v) for n, v in re.findall(r'e\[e\.(\w+)=(\d+)\]=', seg)}
first = s.rfind('[new d(', 0, s.index('new d("3421096c1986d008da88b5fac64cd4c475603138c9bf8a98ab6d581dda6befa7"'))
info = {}
for c in s[first + 1:].split('new d("')[1:]:
    mm = re.match(r'([0-9a-f]{64})",r\.A\.(\w+),s\.A\.(\w+),', c)
    if not mm: break
    fp = re.search(r',c,(\[\[\[.*?\]\]\]|null)', c)
    pid = enum[mm[3]]
    info[pid] = {'cat': mm[2], 'type': mm[3], 'tiles': json.loads(fp[1]) if fp and fp[1] != 'null' else []}
json.dump(info, open('data/part_info.json', 'w'))
print(len(enum), len(info), {k: info[k]['type'] for k in (0, 5, 6, 52, 65) if k in info})
