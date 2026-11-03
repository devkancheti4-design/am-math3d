"""Read members of a remote zip over HTTP Range requests (no full download)."""
import sys, io, json, zipfile, urllib.request

class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0; self.fetched = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r: data = r.read()
        self.pos += len(data); self.fetched += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

if __name__ == "__main__":
    url, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    f = HTTPFile(url)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = z.namelist()
    if mode == "list":
        json.dump(names, open(out, "w"))
        print(len(names), "members; fetched", round(f.fetched / 1e6, 1), "MB of", round(f.size / 1e9, 1), "GB")
    else:
        want = json.load(open(mode))
        import os
        for nm in want:
            data = z.read(nm)
            p = os.path.join(out, nm); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(data)
        print("extracted", len(want), "files; fetched", round(f.fetched / 1e6, 1), "MB")
