"""Check shareable files for common personal paths and credential patterns."""

import gzip
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".jupyter",
    "build",
    "dist",
}
GENERATED = {
    "data/processed",
    "reports/results",
    "reports/figures",
    "reports/executed_notebooks",
}
PATTERNS = {
    "home-directory path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "Windows home-directory path": re.compile(
        r"[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+", re.IGNORECASE
    ),
    "system temporary path": re.compile(r"/(?:private/)?var/(?:folders|tmp)/[A-Za-z0-9]"),
    "operating-system fingerprint": re.compile(r"(?:macOS|Darwin)-\d+\.\d+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "cloud access key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "GitHub access token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "authenticated URL": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}


def shareable(path):
    relative = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in relative.parts):
        return False
    name = relative.as_posix()
    if any(name.startswith(folder + "/") for folder in GENERATED):
        return False
    if name.startswith("artifacts/") and name != "artifacts/README.md":
        return False
    return name != "reports/RESULTS.md"


def contents(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                payload = archive.read(member)
                if member.endswith(".gz"):
                    payload = gzip.decompress(payload)
                yield member, payload.decode("utf-8", errors="replace")
    elif path.suffix == ".gz":
        yield "", gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    else:
        yield "", path.read_bytes().decode("utf-8", errors="replace")


def main():
    issues, count = [], 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or not shareable(path):
            continue
        count += 1
        relative = path.relative_to(ROOT).as_posix()
        if path.name.startswith(".env") or path.suffix in {".pem", ".key"}:
            issues.append(f"{relative}: local configuration or key file")
        if path.name in {".DS_Store", "Thumbs.db"} or path.name.startswith("._"):
            issues.append(f"{relative}: filesystem metadata")
        for member, text in contents(path):
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    suffix = f" ({member})" if member else ""
                    issues.append(f"{relative}{suffix}: {label}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"Checked {count} shareable files; no matching personal paths or credentials found.")


if __name__ == "__main__":
    main()
