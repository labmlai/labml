from pathlib import Path

from labml import monit


def download_file(url: str, path: Path):
    if path.exists():
        return

    import urllib.request

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with monit.section("Download"):
        def reporthook(count, block_size, total_size):
            monit.progress(count * block_size / total_size)

        urllib.request.urlretrieve(url, path, reporthook=reporthook)
