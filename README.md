# mins-datasets
Mirrors multisensor SLAM benchmark datasets onto Hugging Face in a form MINS can run

## NCLT benchmark

NCLT is run straight off its own bucket, nothing mirrored: `nclt.py --stream` turns a session (IMU, wheels,
GPS, Velodyne, Ladybug cameras 1-5) into an unindexed bag on stdout, and `mins_bag` reads it from a pipe.
Ladybug images are fetched one by one through a per-session index of the image archive (`gzidx.c`, `lb3.py`,
built once by `nclt-index.yml` into the release `nclt-lb3-index`).

- `configs/nclt/`: MINS config. Extrinsics from the NCLT paper (Tables 4-5), camera intrinsics fitted as
  equidistant fisheye to the NCLT undistortion maps, wheel track width fitted from the data.
- `bench.sh <mins_bag> <session> <out> <sets...>`: one stream, one `mins_bag` per sensor set, scored by
  `nclt_ate.py` (position ATE after SE3 alignment, m and m/km).
- `nclt-bench.yml`: all sessions on Actions, with a table in the run summary.
