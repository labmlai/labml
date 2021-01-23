from pathlib import Path

from labml import monit


def download_file(url: str, path: Path):
    if path.exists():
        return

    import urllib.request

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with monit.section("Download") as s:
        def reporthook(count, block_size, total_size):
            s.progress(count * block_size / total_size)

        urllib.request.urlretrieve(url, path, reporthook=reporthook)
