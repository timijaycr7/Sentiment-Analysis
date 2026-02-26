def what(file, h=None):
    data = h
    if data is None:
        if isinstance(file, (str, bytes)):
            with open(file, "rb") as stream:
                data = stream.read(32)
        else:
            current_pos = file.tell()
            data = file.read(32)
            file.seek(current_pos)

    for test in TESTS:
        result = test(data)
        if result is not None:
            return result
    return None


def _test_jpeg(data):
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return None


def _test_png(data):
    if data[:8] == b"\211PNG\r\n\032\n":
        return "png"
    return None


def _test_gif(data):
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _test_bmp(data):
    if data[:2] == b"BM":
        return "bmp"
    return None


def _test_webp(data):
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _test_tiff(data):
    if data[:4] in (b"MM\x00*", b"II*\x00"):
        return "tiff"
    return None


TESTS = (
    _test_jpeg,
    _test_png,
    _test_gif,
    _test_bmp,
    _test_webp,
    _test_tiff,
)
