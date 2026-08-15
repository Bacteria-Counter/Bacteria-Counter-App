"""Where does sam_micro's escalation actually fire, and does it help there?

sam_micro is sam_tuned plus a second pass at imgsz 4480, taken when the median
colony is smaller than 1.5% of the dish radius. Aggregate MAE says the two are
close (5.23 against 5.37 on 126 bright plates), but an aggregate cannot say
whether that gap is escalation firing rarely and badly or often and mildly --
and those imply opposite advice.

So: run both over the same plates, keep only the ones where they disagree, and
show the ground truth next to each. A plate the escalation moved AWAY from
truth is the cost; one it moved TOWARD truth is the reason the option exists.
"""
import os
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
MODELS = ROOT / "coreml_models_block32"
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
BRIGHT = ROOT / "bench_data/pca_images"


def run(files, key):
    cmd = [str(SWIFT), str(MODELS)] + [str(f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    res = {}
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) >= 5 and p[1].lstrip("-").isdigit():
            res[p[0]] = (int(p[1]), p[3] == "esk", float(p[4]))
    return res


def main():
    bench = json.load(open(ROOT / "bench_data/pca_benchmark.json"))
    items = [(e["base"], BRIGHT / f"{e['base']}.jpg", e["truth"]) for e in bench]
    items = [i for i in items if i[1].exists()]
    files = [f for _, f, _ in items]

    tuned = run(files, "sam_tuned")
    micro = run(files, "sam_micro")

    rows = []
    for base, f, truth in items:
        if f.name not in tuned or f.name not in micro:
            continue
        t, m = tuned[f.name][0], micro[f.name][0]
        rows.append((base, truth, t, m, micro[f.name][1]))

    fired = [r for r in rows if r[4]]
    changed = [r for r in rows if r[2] != r[3]]
    print(f"{len(rows)} plat terang. Eskalasi menyala di {len(fired)}, "
          f"mengubah hitungan di {len(changed)}.\n")

    if changed:
        print(f"{'plat':<30}{'benar':>7}{'tuned':>8}{'micro':>8}   arah")
        print("-" * 66)
        better = worse = 0
        for base, truth, t, m, _ in sorted(changed, key=lambda r: -abs(r[3] - r[2])):
            dt, dm = abs(t - truth), abs(m - truth)
            mark = "lebih dekat" if dm < dt else ("lebih jauh" if dm > dt else "sama")
            better += dm < dt
            worse += dm > dt
            print(f"{base[:29]:<30}{truth:>7}{t:>8}{m:>8}   {mark}")
        print(f"\nmembaik {better}, memburuk {worse}")

    tt = np.median([v[2] for v in tuned.values()])
    mm = np.median([v[2] for v in micro.values()])
    print(f"\nwaktu median per foto: sam_tuned {tt:.2f}s | sam_micro {mm:.2f}s")
    esc = [micro[f.name][2] for _, f, _ in items
           if f.name in micro and micro[f.name][1]]
    if esc:
        print(f"pada plat yang naik resolusi: {np.median(esc):.2f}s")


if __name__ == "__main__":
    main()
