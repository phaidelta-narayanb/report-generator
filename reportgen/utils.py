
from datetime import datetime
from PIL import Image
import io
from base64 import b64encode


def get_date_string(date_format: str = "%B %d, %Y") -> str:
    return datetime.today().strftime(date_format)

def image2b64(img: Image.Image):
    with io.BytesIO() as f:
        img.save(f, format="png")
        return (b"data:image/png;base64,%s" % b64encode(f.getbuffer())).decode()
