"""Build one NCLT session as a ROS1 bag + TUM ground truth from the raw files on the NCLT S3 bucket.

The official converter (nclt2ros-era sensordata_to_rosbag.py) writes Float64MultiArray, which nothing
reads, so this writes standard messages instead. Hokuyo is left out. Ladybug images are optional and
fetched image by image from the archive (lb3.py), so the bag can be streamed into mins_bag without touching disk.
"""
import argparse, csv, heapq, math, os, struct, subprocess
import lz4.frame
import numpy as np
import rosbags.rosbag1.writer
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

import lb3

BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
ts = get_typestore(Stores.ROS1_NOETIC)
T = ts.types
IMU, FIX, JOINT, PC2 = "sensor_msgs/msg/Imu", "sensor_msgs/msg/NavSatFix", "sensor_msgs/msg/JointState", "sensor_msgs/msg/PointCloud2"
CIMG = "sensor_msgs/msg/CompressedImage"
# rosbags 0.11.5 writes LZ4 frames without a content checksum, which ROS roslz4 rejects ("malformed data").
# Use the frame settings roslz4 itself writes so rosbag play/info read the bag.
rosbags.rosbag1.writer.lz4_compress = lambda x, level=0: lz4.frame.compress(
    x, level, block_size=lz4.frame.BLOCKSIZE_MAX1MB, block_linked=False, content_checksum=True, store_size=False)
HIT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])


def header(utime, frame):
    return T["std_msgs/msg/Header"](seq=0, stamp=T["builtin_interfaces/msg/Time"](sec=utime // 1000000, nanosec=utime % 1000000 * 1000), frame_id=frame)


def rows(path):
    with open(path) as f:
        return [[float(x) for x in r] for r in csv.reader(f)]


def imu_msgs(d):
    # ms25.csv: utime, mag xyz, accel xyz (m/s^2), gyro rpy (rad/s)
    v3, q0 = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"](x=0, y=0, z=0, w=1)
    unknown = np.array([-1.0] + [0.0] * 8)  # orientation not provided
    for r in rows(os.path.join(d, "ms25.csv")):
        t = int(r[0])
        yield t, "/imu", IMU, T[IMU](header=header(t, "imu"), orientation=q0, orientation_covariance=unknown,
                                       angular_velocity=v3(x=r[7], y=r[8], z=r[9]), angular_velocity_covariance=np.zeros(9),
                                       linear_acceleration=v3(x=r[4], y=r[5], z=r[6]), linear_acceleration_covariance=np.zeros(9))


def gps_msgs(d, name, topic, frame="gps"):
    # gps.csv: utime, mode, num_sats, lat (rad), lon (rad), alt (m), track, speed. Only 3D fixes (mode 3)
    # carry an altitude, so the rest are dropped. Covariance is zero: NCLT gives none, MINS then uses its config noise.
    status = T["sensor_msgs/msg/NavSatStatus"](status=0, service=1)
    for r in rows(os.path.join(d, name)):
        if r[1] < 3 or not all(map(math.isfinite, r[3:6])):
            continue
        t = int(r[0])
        yield t, topic, FIX, T[FIX](header=header(t, frame), status=status, latitude=math.degrees(r[3]), longitude=math.degrees(r[4]),
                                     altitude=r[5], position_covariance=np.zeros(9), position_covariance_type=0)


def wheel_msgs(d):
    # wheels.csv: utime, left, right, both linear wheel speeds in m/s (not rad/s)
    for r in rows(os.path.join(d, "wheels.csv")):
        t = int(r[0])
        yield t, "/wheels", JOINT, T[JOINT](header=header(t, "base"), name=["left", "right"], position=np.zeros(2),
                                             velocity=np.array(r[1:3]), effort=np.zeros(0))


def lidar_scans(hits_bin):
    """Yield (utime, PointCloud2) per revolution from velodyne_hits.bin, a stream of timestamped packets.

    Points carry their own time (s, relative to the scan stamp = first packet). A revolution ends where the
    packet azimuth wraps; the HDL-32E spins clockwise here, so azimuth falls and jumps back up by ~360 deg.
    """
    fields = [T["sensor_msgs/msg/PointField"](name=n, offset=o, datatype=k, count=1)
              for n, o, k in [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 2), ("ring", 13, 2), ("time", 14, 7)]]
    pt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "u1"), ("ring", "u1"), ("time", "<f4")])
    scan, t0, last_az = [], None, None
    while True:
        h = hits_bin.read(24)
        if len(h) < 24:
            break
        assert struct.unpack("<HHHH", h[:8]) == (44444,) * 4, "lost packet sync in velodyne_hits.bin"
        n, t = struct.unpack("<IQ", h[8:20])
        hits = np.frombuffer(hits_bin.read(8 * n), dtype=HIT)
        x, y = hits["x"] * 0.005 - 100.0, hits["y"] * 0.005 - 100.0
        az = math.degrees(math.atan2(y.mean(), x.mean())) if n else last_az
        if scan and az is not None and last_az is not None and az - last_az > 180:
            yield t0, cloud(scan, t0, pt, fields)
            scan = []
        if not scan:
            t0 = t
        last_az = az
        p = np.empty(n, dtype=pt)
        p["x"], p["y"], p["z"] = x, y, hits["z"] * 0.005 - 100.0
        p["intensity"], p["ring"], p["time"] = hits["i"], hits["l"], (t - t0) * 1e-6
        scan.append(p)
    if scan:
        yield t0, cloud(scan, t0, pt, fields)


def cloud(scan, t0, pt, fields):
    p = np.concatenate(scan)
    return T[PC2](header=header(t0, "velodyne"), height=1, width=len(p), fields=fields, is_bigendian=False,
                  point_step=pt.itemsize, row_step=pt.itemsize * len(p), data=p.view(np.uint8), is_dense=True)


def gt_to_tum(src, dst):
    # groundtruth_<date>.csv: utime, x, y, z, roll, pitch, yaw of the body in a local NED frame
    n = 0
    with open(dst, "w") as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        for r in rows(src):
            if not all(map(math.isfinite, r)):
                continue
            cr, sr, cp, sp, cy, sy = (g(a / 2) for a in r[4:7] for g in (math.cos, math.sin))
            q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
            f.write("%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (r[0] * 1e-6, *r[1:4], *q))
            n += 1
    print(f"ground truth: {n} poses", flush=True)


def messages(seq, d, fetch, unpack, index=None, cams=()):
    """Yield (utime, topic, type, msg) for session seq in time order; writes d/<seq>.txt (ground truth) on the way.

    Images come from the given Ladybug cameras, fetched one by one from the lb3 archive through its index (lb3.py).
    """
    fetch(f"{BASE}/ground_truth/groundtruth_{seq}.csv", os.path.join(d, "gt.csv"))
    gt_to_tum(os.path.join(d, "gt.csv"), os.path.join(d, f"{seq}.txt"))
    fetch(f"{BASE}/sensor_data/{seq}_sen.tar.gz", os.path.join(d, "sen.tar.gz"))
    unpack(os.path.join(d, "sen.tar.gz"), d)

    sen = os.path.join(d, seq)
    small = sorted([*imu_msgs(sen), *gps_msgs(sen, "gps.csv", "/gps/fix"), *gps_msgs(sen, "gps_rtk.csv", "/gps_rtk/fix", "gps_rtk"), *wheel_msgs(sen)],
                   key=lambda m: m[0])
    # Stream the velodyne archive straight into the converter so it never lands on disk; the bag is the only
    # big file. velodyne_hits.bin is the first member, so --occurrence stops before the per-scan files after it.
    # ponytail: no resume mid-stream (a retry would splice bytes), a dropped connection fails the session; rerun it
    curl = subprocess.Popen(["curl", "-fsL", f"{BASE}/velodyne_data/{seq}_vel.tar.gz"], stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-xzO", "--occurrence=1", f"{seq}/velodyne_hits.bin"], stdin=curl.stdout, stdout=subprocess.PIPE, bufsize=64 << 20)
    curl.stdout.close()
    streams = [small, ((t, "/velodyne_points", PC2, msg) for t, msg in lidar_scans(tar.stdout))]
    if cams:
        # The TIFFs are JPEG inside, which cv::imdecode reads, so they go in unchanged as CompressedImage
        url = f"{BASE}/images/{seq}_lb3.tar.gz"
        streams.append((t, f"/cam{c}/image/compressed", CIMG, T[CIMG](header=header(t, f"cam{c}"), format="tiff", data=np.frombuffer(b, np.uint8)))
                       for t, c, b in lb3.images(url, lb3.load_index(index), cams))
    # on equal times the earlier stream goes first, so the small sensors still precede a scan with the same stamp
    yield from heapq.merge(*streams, key=lambda m: m[0])
    # curl exits 23 when tar stops reading early; that is fine as long as tar got the whole member
    if tar.wait() != 0:
        raise SystemExit("tar failed reading velodyne_hits.bin (download dropped?)")
    curl.wait()


def build(seq, d, fetch, unpack, index=None, cams=()):
    """Download session seq into d and write d/<seq>.bag and d/<seq>.txt."""
    bag, counts = os.path.join(d, f"{seq}.bag"), {}
    w = Writer(bag)
    w.set_compression(Writer.CompressionFormat.LZ4)
    with w:
        conns = {}
        for t, topic, kind, msg in messages(seq, d, fetch, unpack, index, cams):
            if topic not in conns:
                conns[topic] = w.add_connection(topic, kind, typestore=ts)
            w.write(conns[topic], t * 1000, ts.serialize_ros1(msg, kind))
            counts[topic] = counts.get(topic, 0) + 1
    print(f"{seq}: {counts}", flush=True)


class StreamBag:
    """A ROS1 bag written strictly in order with no index, so it can go through a pipe into a reader that plays it as
    it arrives (mins_bag reads an unindexed bag in file order). All connections are declared up front."""

    def __init__(self, f, topics):
        self.f, self.ids, self.buf = f, {}, bytearray()
        head = self.fields(op=b"\x03", index_pos=struct.pack("<Q", 0), conn_count=struct.pack("<I", len(topics)), chunk_count=struct.pack("<I", 0))
        f.write(b"#ROSBAG V2.0\n" + self.record(head, b" " * (4096 - len(head))))  # rosbag pads the header to 4 KB
        for i, (topic, kind) in enumerate(topics.items()):
            msgdef, md5 = ts.generate_msgdef(kind, ros_version=1)
            f.write(self.record(self.fields(op=b"\x07", conn=struct.pack("<I", i), topic=topic.encode()),
                                self.fields(topic=topic.encode(), type=kind.replace("/msg/", "/").encode(), md5sum=md5.encode(), message_definition=msgdef.encode())))
            self.ids[topic] = i

    @staticmethod
    def fields(**kv):
        return b"".join(struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v for k, v in kv.items())

    @staticmethod
    def record(header, data):
        return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data

    def write(self, utime, topic, kind, msg):
        stamp = struct.pack("<II", utime // 1000000, utime % 1000000 * 1000)
        self.buf += self.record(self.fields(op=b"\x02", conn=struct.pack("<I", self.ids[topic]), time=stamp), bytes(ts.serialize_ros1(msg, kind)))
        if len(self.buf) > 8 << 20:
            self.flush()

    def flush(self):
        if self.buf:
            self.f.write(self.record(self.fields(op=b"\x05", compression=b"none", size=struct.pack("<I", len(self.buf))), self.buf))
            self.f.flush()
            self.buf = bytearray()


def stream(seq, d, fetch, unpack, out, index=None, cams=()):
    """Write session seq as an unindexed bag to the binary file out (e.g. a pipe into mins_bag), d/<seq>.txt as usual."""
    topics = {"/imu": IMU, "/gps/fix": FIX, "/gps_rtk/fix": FIX, "/wheels": JOINT, "/velodyne_points": PC2,
              **{f"/cam{c}/image/compressed": CIMG for c in cams}}
    bag, counts = StreamBag(out, topics), {}
    for t, topic, kind, msg in messages(seq, d, fetch, unpack, index, cams):
        bag.write(t, topic, kind, msg)
        counts[topic] = counts.get(topic, 0) + 1
    bag.flush()
    print(f"{seq}: {counts}", flush=True)


if __name__ == "__main__":
    # Standalone, for CI that converts on the fly
    from mirror import fetch, unpack
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("dir", help="working dir; gets <session>.txt (and <session>.bag unless --stream)")
    p.add_argument("--stream", action="store_true", help="write the bag to stdout, unindexed, instead of a file")
    p.add_argument("--index", help="gzidx index of the session's lb3 archive (needed for --cams)")
    p.add_argument("--cams", default="", help="Ladybug cameras to include, e.g. 1,2,3,4,5 (0 looks at the sky)")
    a = p.parse_args()
    cams = tuple(int(c) for c in a.cams.split(",") if c)
    if cams and not a.index:
        p.error("--cams needs --index")
    os.makedirs(a.dir, exist_ok=True)
    if a.stream:
        # stdout carries the bag; everything else that would print there (logs, child processes) goes to stderr
        out = os.fdopen(os.dup(1), "wb")
        os.dup2(2, 1)
        stream(a.session, a.dir, fetch, unpack, out, a.index, cams)
        out.close()
    else:
        build(a.session, a.dir, fetch, unpack, a.index, cams)
