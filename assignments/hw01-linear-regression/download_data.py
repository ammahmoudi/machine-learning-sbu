"""Download and verify the original UCI archive, without altering source files."""

from pathlib import Path
import hashlib
from urllib.request import urlopen
import zipfile

URL = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
DESTINATION = Path(__file__).resolve().parents[2] / "data/raw/bike-sharing"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive = DESTINATION / "bike-sharing.zip"
    if not archive.exists():
        with urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("UCI archive changed: review its provenance before accepting a new hash")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        for name in ("day.csv", "hour.csv", "Readme.txt"):
            content = source.read(name)
            target = DESTINATION / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Existing source file differs; refusing to overwrite {name}")
            if not target.exists():
                target.write_bytes(content)
    print(f"Verified original UCI files in {DESTINATION}")


if __name__ == "__main__":
    main()
