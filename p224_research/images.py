from hashlib import sha256
from io import BytesIO

from PIL import Image


def inspect_png(data):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.n_frames != 1:
            raise ValueError("Expected a single PNG frame")
        image.load()
        return {"sha256": sha256(data).hexdigest(), "mimeType": "image/png",
                "width": image.width, "height": image.height, "bytes": len(data),
                "rgb_sha256": sha256(image.convert("RGB").tobytes()).hexdigest()}


def visual(path,h):
    import cv2
    import numpy as np

    with Image.open(path) as image:
        rgb=np.asarray(image.convert("RGB").resize((320,h),Image.Resampling.BILINEAR))
    return rgb,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
