from PIL import Image
from PIL.PngImagePlugin import PngInfo

def read_png_metadata(filepath: str) -> dict:
    """Читает текстовые метаданные из PNG-файла."""
    image = Image.open(filepath)
    metadata = {}

    if image.text:
        for key, value in image.text.items():
            metadata[key] = value
            print(f"{key}: {value}")
    else:
        print("Текстовые метаданные отсутствуют.")

    return metadata


# Использование:
meta = read_png_metadata("png_image_1.png")