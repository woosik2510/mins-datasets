"""Stream Ladybug3 images out of an NCLT lb3 .tar.gz on the bucket in time order, without downloading the archive.

The archive is one gzip stream with images in random order, so it is indexed once (gzidx.c) and each image is then
one HTTP range request plus an inflate from a deflate block boundary. Python's zlib cannot start mid-byte (no
inflatePrime), so libz is called through ctypes; ctypes drops the GIL, so the fetch threads also decode in parallel.
"""
import collections, ctypes, ctypes.util, gzip, http.client, random, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

Entry = collections.namedtuple("Entry", "name upos data size cin cend bits window")


def load_index(path):
    """Index entries (gzidx output, optionally gzipped) by name."""
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        b = f.read()
    out, i = {}, 0
    while i < len(b):
        (n,) = struct.unpack_from("<H", b, i)
        name = b[i + 2:i + 2 + n].decode()
        i += 2 + n
        upos, data, size, cin, cend, bits = struct.unpack_from("<6Q", b, i)
        (k,) = struct.unpack_from("<I", b, i + 48)
        i += 52
        pos = struct.unpack_from(f"<{k}H", b, i)
        val = b[i + 2 * k:i + 3 * k]
        i += 3 * k
        out[name] = Entry(name, upos, data, size, cin, cend, bits, (pos, val))
    return out


class ZStream(ctypes.Structure):
    _fields_ = [("next_in", ctypes.c_void_p), ("avail_in", ctypes.c_uint), ("total_in", ctypes.c_ulong),
                ("next_out", ctypes.c_void_p), ("avail_out", ctypes.c_uint), ("total_out", ctypes.c_ulong),
                ("msg", ctypes.c_char_p), ("state", ctypes.c_void_p), ("zalloc", ctypes.c_void_p),
                ("zfree", ctypes.c_void_p), ("opaque", ctypes.c_void_p), ("data_type", ctypes.c_int),
                ("adler", ctypes.c_ulong), ("reserved", ctypes.c_ulong)]


z = ctypes.CDLL(ctypes.util.find_library("z"))
z.zlibVersion.restype = ctypes.c_char_p


def decode(e, comp):
    """Inflate entry e from comp, the archive bytes [cin, cend)."""
    window = bytearray(32768)
    for p, v in zip(*e.window):
        window[p] = v
    s = ZStream()
    if z.inflateInit2_(ctypes.byref(s), -15, z.zlibVersion(), ctypes.sizeof(s)) != 0:
        raise RuntimeError("inflateInit2 failed")
    try:
        if e.bits:
            z.inflatePrime(ctypes.byref(s), e.bits, comp[0] >> (8 - e.bits))
        z.inflateSetDictionary(ctypes.byref(s), bytes(window), 32768)
        src = ctypes.create_string_buffer(comp[1 if e.bits else 0:])
        n = e.data - e.upos + e.size
        dst = ctypes.create_string_buffer(n)
        s.next_in, s.avail_in = ctypes.addressof(src), len(src) - 1
        s.next_out, s.avail_out = ctypes.addressof(dst), n
        while s.avail_out:
            r = z.inflate(ctypes.byref(s), 0)
            if r not in (0, 1) or (r == 1 and s.avail_out):
                raise RuntimeError(f"inflate failed ({r}) on {e.name}")
        return dst.raw[e.data - e.upos:]
    finally:
        z.inflateEnd(ctypes.byref(s))


def fetch(url, e, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={e.cin}-{e.cend - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                comp = r.read()
            if len(comp) != e.cend - e.cin:
                raise IOError(f"short read {len(comp)}/{e.cend - e.cin}")
            return decode(e, comp)
        except (OSError, http.client.HTTPException) as err:
            if t == tries - 1:
                raise
            print(f"lb3: {e.name}: {err}, retry", file=sys.stderr, flush=True)
            time.sleep(5 * (t + 1))


def images(url, index, cams, workers=32, ahead=256):
    """Yield (utime, cam, tiff bytes) for the given cameras in time order, fetching up to `ahead` images early."""
    todo = []
    for name in index:
        parts = name.split("/")  # <session>/lb3/Cam<k>/<utime>.tiff
        cam = int(parts[-2][3:])
        if cam in cams:
            todo.append((int(parts[-1].split(".")[0]), cam, index[name]))
    todo.sort(key=lambda x: (x[0], x[1]))
    with ThreadPoolExecutor(workers) as ex:
        q = collections.deque()
        for t, cam, e in todo:
            q.append((t, cam, ex.submit(fetch, url, e)))
            if len(q) >= ahead:
                t0, c0, f = q.popleft()
                yield t0, c0, f.result()
        while q:
            t0, c0, f = q.popleft()
            yield t0, c0, f.result()


def cat(url, out, block=8 << 20):
    """Copy url to the binary file out, resuming by range when the connection drops (the index pass reads up to 140 GB)."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    got = fails = 0
    while got < size:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={got}-"}), timeout=60) as r:
                while b := r.read(block):
                    out.write(b)
                    got += len(b)
        except (OSError, http.client.HTTPException) as err:
            fails += 1
            if fails > 30:
                raise
            print(f"lb3: {url} dropped at {got}/{size}: {err}, resuming", file=sys.stderr, flush=True)
            time.sleep(10)
    out.flush()


def check(url, index, n=50):
    """Fetch n random images and check they come out as TIFFs of the indexed size."""
    for e in random.sample(sorted(index.values()), min(n, len(index))):
        b = fetch(url, e)
        if len(b) != e.size or b[:4] not in (b"II*\0", b"MM\0*"):
            raise SystemExit(f"lb3: {e.name} does not decode to a TIFF")
    print(f"lb3: {min(n, len(index))} random images of {len(index)} decode fine", file=sys.stderr)


if __name__ == "__main__":
    # python lb3.py cat URL > archive      python lb3.py check URL INDEX
    if sys.argv[1] == "cat":
        cat(sys.argv[2], sys.stdout.buffer)
    elif sys.argv[1] == "check":
        check(sys.argv[2], load_index(sys.argv[3]))
