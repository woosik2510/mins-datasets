// Index the members of a .tar.gz so each one can be read on its own from a byte range of the archive.
//
//   curl -s URL | gzidx > archive.idx      one streaming pass; lb3.py reads the index and fetches members
//
// A gzip stream can only be decoded from the start, or from a deflate block boundary given the 32 KB of output
// before it (the window). Each member gets the last boundary before its data. Storing the whole window would make
// the index 6% of the archive, so only the window bytes the member actually references are kept (a sparse window,
// as rapidgzip does); on already-compressed payloads like these JPEG TIFFs that is ~100 bytes.
//
// Index entry, little endian: u16 name length, name, u64 upos (boundary, uncompressed), data (member start), size,
// cin (first compressed byte to fetch), cend (end of the range), bits (bits of byte cin used by the previous
// block), u32 n, n x u16 window positions, n x u8 their values.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define WIN 32768
#define IN 65536
#define RING (1u << 26)  // true output kept for looking up window bytes; trackers lag the main stream by < 1 MB

static void die(const char *m) {
    fprintf(stderr, "gzidx: %s\n", m);
    exit(1);
}

static void start_at(z_stream *s, int bits, int byte0, const unsigned char *window) {
    memset(s, 0, sizeof(*s));
    if (inflateInit2(s, -15) != Z_OK) die("inflateInit2");
    if (bits) inflatePrime(s, bits, byte0 >> (8 - bits));
    inflateSetDictionary(s, window, WIN);
}

// A member in progress: three inflaters started at its boundary with made-up windows. Window 0 is all zero, windows
// 1 and 2 hold each byte's own position (low byte, high byte | 0x80). An output byte differs between windows 0 and 2
// exactly when it is (a copy of a copy of ...) a window byte, and windows 1 and 2 then say which one.
typedef struct {
    char name[101];
    uint64_t upos, data, size, cin;
    int bits;
    z_stream z[3];
    uint64_t out;
    unsigned char ref[WIN];
} Track;

static unsigned char zeros[WIN], marks[2][WIN], *ring;

static void track_feed(Track *t, unsigned char *in, size_t n) {
    static unsigned char buf[3][1 << 16];
    uint64_t end = t->data + t->size;
    for (int w = 0; w < 3; w++) t->z[w].next_in = in, t->z[w].avail_in = (uInt)n;
    while (t->out < end) {
        // never decode past the member, so the input consumed marks where its range ends
        uInt room = end - t->out < sizeof(buf[0]) ? (uInt)(end - t->out) : (uInt)sizeof(buf[0]);
        size_t got[3];
        for (int w = 0; w < 3; w++) {
            t->z[w].next_out = buf[w];
            t->z[w].avail_out = room;
            int r = inflate(&t->z[w], Z_NO_FLUSH);
            if (r != Z_OK && r != Z_STREAM_END && r != Z_BUF_ERROR) die("inflate failed in a member");
            got[w] = room - t->z[w].avail_out;
        }
        if (got[1] != got[0] || got[2] != got[0]) die("trackers out of step");
        for (size_t i = 0; i < got[0]; i++)
            if (t->out + i >= t->data && buf[0][i] != buf[2][i]) t->ref[buf[1][i] | (buf[2][i] & 0x7f) << 8] = 1;
        t->out += got[0];
        if (!got[0]) break;  // needs more input
    }
}

static void track_done(Track *t) {
    uint16_t n = (uint16_t)strlen(t->name);
    uint64_t v[6] = {t->upos, t->data, t->size, t->cin, t->cin + (t->bits ? 1 : 0) + t->z[0].total_in, (uint64_t)t->bits};
    uint16_t pos[WIN];
    unsigned char val[WIN];
    uint32_t k = 0;
    for (int i = 0; i < WIN; i++)
        if (t->ref[i]) {
            if (t->upos < (uint64_t)(WIN - i)) die("reference before the stream start");
            pos[k] = (uint16_t)i;
            val[k++] = ring[(t->upos - WIN + i) % RING];
        }
    fwrite(&n, 2, 1, stdout);
    fwrite(t->name, 1, n, stdout);
    fwrite(v, 8, 6, stdout);
    fwrite(&k, 4, 1, stdout);
    fwrite(pos, 2, k, stdout);
    fwrite(val, 1, k, stdout);
    for (int w = 0; w < 3; w++) inflateEnd(&t->z[w]);
}

int main(void) {
    ring = malloc(RING);
    for (int i = 0; i < WIN; i++) marks[0][i] = (unsigned char)i, marks[1][i] = (unsigned char)(i >> 8 | 0x80);

    unsigned char h[10];
    if (fread(h, 1, 10, stdin) != 10 || h[0] != 0x1f || h[1] != 0x8b || h[2] != 8) die("not gzip");
    uint64_t cpos = 10;  // compressed bytes read so far
    if (h[3] & 4) {
        int x = getchar();
        x |= getchar() << 8;
        for (cpos += 2 + x; x--;) getchar();
    }
    for (int fl = 8; fl <= 16; fl <<= 1)
        if (h[3] & fl)
            for (cpos++; getchar() > 0;) cpos++;
    if (h[3] & 2) getchar(), getchar(), cpos += 2;

    z_stream z;
    memset(&z, 0, sizeof(z));
    if (inflateInit2(&z, -15) != Z_OK) die("inflateInit2");
    // compressed bytes from the oldest point still needed (current block boundary or a running tracker) on
    size_t hcap = 1 << 22, hlen = 0;
    unsigned char *hist = malloc(hcap);
    uint64_t hist_c0 = cpos;
    uint64_t out = 0, blk_u = 0, blk_c = cpos;  // current block boundary: uncompressed, compressed byte holding it
    int blk_bits = 0;
    uint64_t next_hdr = 0, want = UINT64_MAX;  // tar: next header offset; data start of a member not yet tracked
    unsigned char hdr[512];
    int hdr_have = 0;
    Track pend, **tr = NULL;
    int ntr = 0, ctr = 0;
    uint64_t members = 0, refs = 0;
    static unsigned char in[IN], ob[1 << 16];
    for (int done = 0; !done;) {
        size_t n = fread(in, 1, IN, stdin);
        if (!n) break;
        if (hlen + n > hcap) hist = realloc(hist, hcap = 2 * (hlen + n));
        memcpy(hist + hlen, in, n);
        hlen += n;
        z.next_in = in;
        z.avail_in = (uInt)n;
        while (z.avail_in && !done) {
            z.next_out = ob;
            z.avail_out = sizeof(ob);
            int r = inflate(&z, Z_BLOCK);  // stops at every block boundary
            if (r != Z_OK && r != Z_STREAM_END && r != Z_BUF_ERROR) die("inflate failed");
            size_t got = sizeof(ob) - z.avail_out;
            for (size_t i = 0; i < got; i++) ring[(out + i) % RING] = ob[i];
            for (size_t i = 0; i < got;) {  // tar headers in this output
                if (out + i < next_hdr) {
                    uint64_t skip = next_hdr - (out + i);
                    i += skip < got - i ? skip : got - i;
                    continue;
                }
                size_t take = 512 - hdr_have < got - i ? 512 - hdr_have : got - i;
                memcpy(hdr + hdr_have, ob + i, take);
                hdr_have += (int)take;
                i += take;
                if (hdr_have < 512) break;
                hdr_have = 0;
                if (!hdr[0]) {  // end-of-archive block
                    next_hdr = UINT64_MAX;
                    continue;
                }
                char sz[13] = {0};
                memcpy(sz, hdr + 124, 12);
                uint64_t size = strtoull(sz, NULL, 8), data = next_hdr + 512;
                next_hdr = data + (size + 511) / 512 * 512;
                if ((hdr[156] == '0' || hdr[156] == 0) && size) {
                    if (want != UINT64_MAX) die("two members pending");
                    memset(&pend, 0, sizeof(pend));
                    memcpy(pend.name, hdr, 100);
                    pend.data = want = data;
                    pend.size = size;
                }
            }
            uint64_t consumed = cpos + (n - z.avail_in);
            if (want != UINT64_MAX && want < out + got) {
                // the member starts in this block: track it from the block's boundary, replaying what came since
                if (want < blk_u) die("missed a member start");
                if (ntr == ctr) tr = realloc(tr, sizeof(Track *) * (ctr = ctr ? 2 * ctr : 8));
                Track *t = tr[ntr++] = malloc(sizeof(Track));
                *t = pend;
                t->upos = t->out = blk_u;
                t->cin = blk_c;
                t->bits = blk_bits;
                size_t off = (size_t)(blk_c - hist_c0), from = off + (blk_bits ? 1 : 0);
                for (int w = 0; w < 3; w++) start_at(&t->z[w], blk_bits, blk_bits ? hist[off] : 0, w ? marks[w - 1] : zeros);
                track_feed(t, hist + from, (size_t)(consumed - hist_c0) - from);
                want = UINT64_MAX;
            }
            out += got;
            if (r == Z_STREAM_END) done = 1;
            if ((z.data_type & 128) && !(z.data_type & 64)) {
                blk_bits = z.data_type & 7;
                blk_c = consumed - (blk_bits ? 1 : 0);
                blk_u = out;
            }
        }
        cpos += n;
        // bring every tracker up to the end of this chunk; write out the ones whose member is complete
        uint64_t need = blk_c;
        for (int k = 0; k < ntr;) {
            Track *t = tr[k];
            uint64_t tc = t->cin + (t->bits ? 1 : 0) + t->z[0].total_in;
            if (tc < cpos) track_feed(t, hist + (tc - hist_c0), (size_t)(cpos - tc));
            if (t->out >= t->data + t->size) {
                track_done(t);
                members++;
                for (int i = 0; i < WIN; i++) refs += t->ref[i];
                free(t);
                tr[k] = tr[--ntr];
                continue;
            }
            tc = t->cin + (t->bits ? 1 : 0) + t->z[0].total_in;
            if (tc < need) need = tc;
            k++;
        }
        if (need > hist_c0) {
            memmove(hist, hist + (need - hist_c0), hlen - (size_t)(need - hist_c0));
            hlen -= (size_t)(need - hist_c0);
            hist_c0 = need;
        }
    }
    fprintf(stderr, "gzidx: %llu members, %.0f window bytes each, %llu bytes decompressed%s\n", (unsigned long long)members,
            members ? (double)refs / members : 0.0, (unsigned long long)out, ntr ? ", archive truncated" : "");
    return 0;
}
