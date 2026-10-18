#!/usr/bin/env python3
"""Mirror benchmark datasets onto Hugging Face in the shape MINS CI reads: <seq>.bag + <seq>.txt (TUM ground truth).

  mirror.py list <dataset> [only]   JSON list of sequence names, for the workflow matrix
  mirror.py seq  <dataset> <seq>    download, unpack, convert ground truth, upload, clean up
  mirror.py card <dataset>          write the dataset README on Hugging Face
"""
import fnmatch, glob, json, os, shutil, subprocess, sys
import yaml
from huggingface_hub import HfApi

api = HfApi()


def manifest(name):
    with open(f"datasets/{name}.yaml") as f:
        m = yaml.safe_load(f)
    # A sequence is either a bare name, filled in from the dataset-level url templates, or a dict of its own
    m["seqs"] = [x if isinstance(x, dict) else {"name": x} for x in m["seqs"]]
    for x in m["seqs"]:
        for key in ("bag", "gt"):
            if key not in x and f"{key}_url" in m:
                x[key] = m[f"{key}_url"].format(name=x["name"])
    return m


def workdir():
    # Runners have a big root disk and a smaller /mnt; take whichever has more room
    best = max(["/mnt", os.path.expanduser("~")], key=lambda d: shutil.disk_usage(d).free if os.access(d, os.W_OK) else 0)
    d = os.path.join(best, "mirror")
    os.makedirs(d, exist_ok=True)
    return d


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(url, dst):
    """Download url to dst. Schemes: http(s), gdrive:<file id>, s3://bucket/key (public), hf://<repo>/<path>."""
    if url.startswith("gdrive:"):
        run("gdown", "--continue", "-O", dst, url[len("gdrive:"):])
    elif url.startswith("s3://"):
        run("aws", "s3", "cp", "--no-sign-request", "--only-show-errors", url, dst)
    elif url.startswith("hf://"):
        repo, path = url[5:].split("/", 2)[:2], url[5:].split("/", 2)[2]
        run("hf", "download", "/".join(repo), path, "--repo-type", "dataset", "--local-dir", os.path.dirname(dst))
        os.replace(os.path.join(os.path.dirname(dst), path), dst)
    else:
        # -C - resumes a dropped transfer from what already landed instead of starting over
        run("curl", "-fL", "--retry", "20", "--retry-all-errors", "--retry-delay", "10", "-C", "-", "-sS", "-o", dst, url)


def unpack(archive, into):
    if archive.endswith(".zip"):
        run("unzip", "-q", "-o", archive, "-d", into)
    elif archive.endswith((".7z", ".7z.001")):
        run("7z", "x", "-y", f"-o{into}", archive)
    elif archive.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
        run("tar", "-xf", archive, "-C", into)
    else:
        raise SystemExit(f"unknown archive type: {archive}")
    os.remove(archive)


def to_tum(src, dst, fmt):
    """Rewrite ground truth as 'timestamp(s) tx ty tz qx qy qz qw'."""
    rows = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%t" or line[0].isalpha():
                continue
            v = [float(x) for x in line.replace(",", " ").split()]
            if fmt == "tum":  # t x y z qx qy qz qw
                t, p, q = v[0], v[1:4], v[4:8]
            elif fmt == "euroc":  # t[ns] x y z qw qx qy qz ...
                t, p, q = v[0], v[1:4], [v[5], v[6], v[7], v[4]]
            elif fmt == "tum_wxyz":  # t x y z qw qx qy qz
                t, p, q = v[0], v[1:4], [v[5], v[6], v[7], v[4]]
            else:
                raise SystemExit(f"unknown ground truth format: {fmt}")
            rows.append((t, p, q))
    # Timestamps in ns (or us) are far above any plausible epoch in seconds
    scale = 1e-9 if rows and rows[0][0] > 1e17 else 1e-6 if rows and rows[0][0] > 1e14 else 1.0
    with open(dst, "w") as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        for t, p, q in rows:
            f.write("%.9f %s %s\n" % (t * scale, " ".join("%.9f" % x for x in p), " ".join("%.9f" % x for x in q)))
    print(f"ground truth: {len(rows)} poses, scale {scale}", flush=True)


def gt_from_bag(bag, topic, dst):
    """Write the poses on one topic of the bag as TUM ground truth."""
    from pathlib import Path
    from rosbags.highlevel import AnyReader

    n = 0
    with AnyReader([Path(bag)]) as reader, open(dst, "w") as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"no topic {topic} in {bag}")
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if hasattr(msg, "transform"):  # TransformStamped
                p, q = msg.transform.translation, msg.transform.rotation
            elif hasattr(msg.pose, "pose"):  # Odometry, PoseWithCovarianceStamped
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
            else:  # PoseStamped
                p, q = msg.pose.position, msg.pose.orientation
            f.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            n += 1
    print(f"ground truth: {n} poses from {topic}", flush=True)


def cmd_list(name, only=""):
    seqs = [s["name"] for s in manifest(name)["seqs"]]
    if only:
        want = only.split(",")
        seqs = [s for s in seqs if s in want]
    print("seqs=" + json.dumps(seqs))


def cmd_seq(name, seq):
    m = manifest(name)
    s = next(x for x in m["seqs"] if x["name"] == seq)
    repo = m["hf_repo"]
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    have = set(api.list_repo_files(repo, repo_type="dataset"))
    if f"{seq}.bag" in have and f"{seq}.txt" in have:
        print(f"{seq}: already on {repo}, skipping")
        return

    d = os.path.join(workdir(), seq)
    os.makedirs(d, exist_ok=True)
    bag, gt = os.path.join(d, f"{seq}.bag"), os.path.join(d, f"{seq}.txt")
    if m.get("convert"):
        # Raw-file datasets: a module named after the converter builds the bag and ground truth itself
        __import__(m["convert"]).build(seq, d, fetch, unpack)
    else:
        # A bag may come as one file, as split parts to concatenate, or inside an archive
        parts = s["bag"] if isinstance(s["bag"], list) else [s["bag"]]
        files = []
        for i, url in enumerate(parts):
            dst = os.path.join(d, s.get("files", [None] * len(parts))[i] or f"part{i}" + os.path.splitext(url.split("?")[0])[1])
            fetch(url, dst)
            files.append(dst)
        if s.get("archive"):
            unpack(files[0], d)
            hits = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True) if fnmatch.fnmatch(os.path.basename(p), s.get("bag_in_archive", "*.bag"))]
            if len(hits) != 1:
                raise SystemExit(f"expected one bag in the archive, found {hits}")
            os.replace(hits[0], bag)
        elif len(files) > 1:
            with open(bag, "wb") as out:
                for p in files:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out, 64 << 20)
                    os.remove(p)
        else:
            os.replace(files[0], bag)

        fmt = s.get("gt_format", m.get("gt_format", "tum"))
        gt_raw = os.path.join(d, "gt_raw")
        if fmt.startswith("bag:"):
            gt_from_bag(bag, fmt[len("bag:"):], gt)
        elif s.get("gt_in_archive"):
            hits = glob.glob(os.path.join(d, "**", s["gt_in_archive"]), recursive=True)
            if len(hits) != 1:
                raise SystemExit(f"expected one ground truth file in the archive, found {hits}")
            os.replace(hits[0], gt_raw)
        else:
            fetch(s["gt"], gt_raw)
        if not fmt.startswith("bag:"):
            to_tum(gt_raw, gt, fmt)

    size = os.path.getsize(bag)
    if size > 190e9:
        raise SystemExit(f"{seq}.bag is {size / 1e9:.0f} GB, over the 200 GB Hugging Face file limit; split it first")
    api.upload_file(path_or_fileobj=gt, path_in_repo=f"{seq}.txt", repo_id=repo, repo_type="dataset", commit_message=f"add {seq} ground truth")
    api.upload_file(path_or_fileobj=bag, path_in_repo=f"{seq}.bag", repo_id=repo, repo_type="dataset", commit_message=f"add {seq}")
    shutil.rmtree(d)
    print(f"{seq}: uploaded {size / 1e9:.1f} GB to {repo}")


def cmd_card(name):
    m = manifest(name)
    seqs = "\n".join(f"- `{s['name']}`" for s in m["seqs"])
    topics = "\n".join(f"- {k}: `{v}`" for k, v in m.get("topics", {}).items())
    card = f"""---
license: {m.get('hf_license', 'other')}
pretty_name: {m['title']}
tags: [robotics, slam, ros, multisensor]
---

# {m['title']} (mirror)

Mirror of [{m['title']}]({m['source']}), repackaged for [MINS](https://github.com/rpng/MINS) evaluation.
All credit belongs to the original authors; please cite them (below) if you use this data.

**License:** {m['license']}

Each sequence is `<seq>.bag` (ROS1, {m.get('bag_note', 'unchanged from the original')}) plus `<seq>.txt`, the ground truth
rewritten as `timestamp(s) tx ty tz qx qy qz qw`. {m.get('notes', '')}

## Sequences
{seqs}

## Topics
{topics}

## Citation
```bibtex
{m['citation'].strip()}
```
"""
    api.create_repo(m["hf_repo"], repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=m["hf_repo"], repo_type="dataset", commit_message="update card")
    print(card)


if __name__ == "__main__":
    {"list": cmd_list, "seq": cmd_seq, "card": cmd_card}[sys.argv[1]](*sys.argv[2:])
