"""Score a MINS trajectory against NCLT ground truth: position ATE after SE3 alignment, and m per km of GT path.

  nclt_ate.py <gt.txt> <traj.txt> <session> <set> [exit code]   -> one JSON line
  nclt_ate.py report <dir with */score.jsonl>                    -> markdown table

MINS writes the IMU pose and the ground truth is the body (axle centre) pose; the 0.21 m horizontal lever arm
between them is ignored (ponytail: well under the errors here, apply it if results get to that level).
"""
import glob, json, os, sys
import numpy as np


def ate_se3(gt, est):
    idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
    idx -= (est[:, 0] - gt[idx - 1, 0]) < (gt[idx, 0] - est[:, 0])
    ok = np.abs(gt[idx, 0] - est[:, 0]) < 0.05
    p_e, p_g = est[ok, 1:4], gt[idx[ok], 1:4]
    ce, cg = p_e.mean(0), p_g.mean(0)
    U, _, Vt = np.linalg.svd((p_g - cg).T @ (p_e - ce))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    err = (p_e - ce) @ R.T + cg - p_g
    return float(np.sqrt((err ** 2).sum(1).mean())), int(ok.sum())


def score(gt_path, est_path, seq, combo, code):
    gt = np.loadtxt(gt_path, usecols=range(4))
    km = float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum() / 1000)
    res = {"seq": seq, "set": combo, "km": round(km, 3), "exit": code}
    try:
        est = np.loadtxt(est_path, usecols=range(4), ndmin=2)
        ate, n = ate_se3(gt, est)
        # a run that died early scores well on its short prefix, so report coverage too
        cover = float((est[-1, 0] - gt[0, 0]) / (gt[-1, 0] - gt[0, 0]))
        res.update(ate=round(ate, 3), m_km=round(ate / km, 3), poses=n, cover=round(cover, 3))
    except Exception as e:
        res["error"] = str(e)[:80]
    print(json.dumps(res), flush=True)


def report(d):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "*", "score.jsonl"))) for l in open(f) if l.strip()]
    sets = sorted({r["set"] for r in rows}, key=lambda s: (len(s), s))
    by = {(r["seq"], r["set"]): r for r in rows}
    print("ATE after SE3 alignment, m per km of ground-truth path (absolute m in brackets). "
          "Runs that stopped before 98% of the session show their coverage.\n")
    print("| session | km | " + " | ".join(sets) + " |")
    print("|---|---|" + "---|" * len(sets))
    for seq in sorted({r["seq"] for r in rows}):
        cells = []
        for s in sets:
            r = by.get((seq, s), {})
            if "m_km" not in r:
                cells.append("fail")
            elif r["cover"] < 0.98:
                cells.append("stopped at %d%%" % (100 * r["cover"]))
            else:
                cells.append("%.2f (%.1f)" % (r["m_km"], r["ate"]))
        km = next(r["km"] for r in rows if r["seq"] == seq)
        print("| %s | %.1f | %s |" % (seq, km, " | ".join(cells)))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        score(*sys.argv[1:5], sys.argv[5] if len(sys.argv) > 5 else None)
