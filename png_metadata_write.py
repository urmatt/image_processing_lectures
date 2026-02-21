from PIL import Image
from PIL.PngImagePlugin import PngInfo

def write_png_metadata(filepath: str, output_path: str, metadata: dict):
    """Записывает текстовые метаданные в PNG-файл.

    Args:
        filepath: Путь к исходному PNG-файлу.
        output_path: Путь для сохранения результата.
        metadata: Словарь {ключ: значение} с метаданными.
    """
    image = Image.open(filepath)
    png_info = PngInfo()

    for key, value in metadata.items():
        # add_text — для tEXt (Latin-1)
        # add_itxt — для iTXt (UTF-8, поддержка Unicode)
        png_info.add_itxt(key, value, lang="ru", tkey=key)

    image.save(output_path, pnginfo=png_info)
    print(f"Метаданные записаны в {output_path}")


# Использование:
write_png_metadata("input.png", "output.png", {
    "Title": "Пример изображения",
    "Author": "Иван Петров",
    "Description": "Демонстрационное изображение для лекции",
    "Copyright": "© 2024 Иван Петров",
    "Creation Time": "2024-03-15T14:30:00+06:00",
    "Software": "Python Pillow",
    "Comment": "Это комментарий на русском языке"
})