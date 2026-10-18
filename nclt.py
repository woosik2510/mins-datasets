"""Build one NCLT session as a ROS1 bag + TUM ground truth from the raw files on the NCLT S3 bucket.

The official converter (nclt2ros-era sensordata_to_rosbag.py) writes Float64MultiArray, which nothing
reads, so this writes standard messages instead. Ladybug images (~114 GB/session) and Hokuyo are left out.
"""
import csv, math, os, struct, subprocess
import lz4.frame
import numpy as np
import rosbags.rosbag1.writer
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
ts = get_typestore(Stores.ROS1_NOETIC)
T = ts.types
IMU, FIX, JOINT, PC2 = "sensor_msgs/msg/Imu", "sensor_msgs/msg/NavSatFix", "sensor_msgs/msg/JointState", "sensor_msgs/msg/PointCloud2"
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


def build(seq, d, fetch, unpack):
    """Download session seq into d and write d/<seq>.bag and d/<seq>.txt."""
    fetch(f"{BASE}/ground_truth/groundtruth_{seq}.csv", os.path.join(d, "gt.csv"))
    gt_to_tum(os.path.join(d, "gt.csv"), os.path.join(d, f"{seq}.txt"))
    fetch(f"{BASE}/sensor_data/{seq}_sen.tar.gz", os.path.join(d, "sen.tar.gz"))
    unpack(os.path.join(d, "sen.tar.gz"), d)
    vel = os.path.join(d, "vel.tar.gz")
    fetch(f"{BASE}/velodyne_data/{seq}_vel.tar.gz", vel)

    sen = os.path.join(d, seq)
    small = sorted([*imu_msgs(sen), *gps_msgs(sen, "gps.csv", "/gps/fix"), *gps_msgs(sen, "gps_rtk.csv", "/gps_rtk/fix", "gps_rtk"), *wheel_msgs(sen)],
                   key=lambda m: m[0])
    # velodyne_hits.bin is the first member, so --occurrence stops tar before it reads the per-scan files after it
    tar = subprocess.Popen(["tar", "-xzOf", vel, "--occurrence=1", f"{seq}/velodyne_hits.bin"], stdout=subprocess.PIPE, bufsize=64 << 20)
    bag, counts, k = os.path.join(d, f"{seq}.bag"), {}, 0
    w = Writer(bag)
    w.set_compression(Writer.CompressionFormat.LZ4)
    with w:
        conns = {}

        def put(t, topic, kind, msg):
            if topic not in conns:
                conns[topic] = w.add_connection(topic, kind, typestore=ts)
            w.write(conns[topic], t * 1000, ts.serialize_ros1(msg, kind))
            counts[topic] = counts.get(topic, 0) + 1

        # Write in time order so a reader that streams by connection index sees a consistent bag
        for t, msg in lidar_scans(tar.stdout):
            while k < len(small) and small[k][0] <= t:
                put(*small[k])
                k += 1
            put(t, "/velodyne_points", PC2, msg)
        for m in small[k:]:
            put(*m)
    if tar.wait() != 0:
        raise SystemExit("tar failed reading velodyne_hits.bin")
    os.remove(vel)
    print(f"{seq}: {counts}", flush=True)
