from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def read_exif(filepath: str) -> dict:
    """Читает EXIF-метаданные из JPEG-файла."""
    image = Image.open(filepath)
    exif_data = image._getexif()

    if exif_data is None:
        print("EXIF-данные отсутствуют.")
        return {}

    metadata = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, f"Unknown({tag_id})")
        metadata[tag_name] = value

    return metadata


# Использование:
meta = read_exif("/Users/urmat/Documents/DIN_1_24/PXL_20260221_030334543_new_meta.jpg")
for key, value in meta.items():
    print(f"{key}: {value}")