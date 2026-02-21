import piexif
from datetime import datetime

def write_exif_to_jpeg(filepath: str, output_path: str):
    """Записывает пользовательские EXIF-данные в JPEG-файл."""

    # Загружаем существующие EXIF или создаём новые
    try:
        exif_dict = piexif.load(filepath)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    # Записываем данные в IFD0
    exif_dict["0th"][piexif.ImageIFD.Make] = "Canon"
    exif_dict["0th"][piexif.ImageIFD.Model] = "X100"
    exif_dict["0th"][piexif.ImageIFD.Software] = "KGIMAGE 1.0"
    exif_dict["0th"][piexif.ImageIFD.Artist] = "Urmat".encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Copyright] = "© 2026 Urmat".encode("utf-8")

    # Записываем данные в Exif Sub IFD
    now = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = now
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = now
    # exif_dict["Exif"][piexif.ExifIFD.UserComment] = "Тестовое фото для лекции"

    # Записываем GPS-данные (Бишкек: 42.8746°N, 74.5698°E)
    lat_deg = _decimal_to_dms(42.8565)
    lon_deg = _decimal_to_dms(74.6660)

    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = lat_deg
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E"
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = lon_deg

    # Сериализуем и сохраняем
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, filepath, new_file=output_path)

    print(f"EXIF-данные записаны в {output_path}")


def _decimal_to_dms(decimal: float) -> tuple:
    """Конвертирует десятичные градусы в формат DMS для EXIF.

    Возвращает tuple из трёх рациональных чисел (градусы, минуты, секунды),
    каждое представленное как tuple (числитель, знаменатель).
    """
    decimal = abs(decimal)
    degrees = int(decimal)
    minutes_float = (decimal - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60 * 10000)

    return (
        (degrees, 1),
        (minutes, 1),
        (seconds, 10000)
    )


# Использование:
write_exif_to_jpeg("PXL_20260221_030334543.jpg", "PXL_20260221_030334543_new_meta.jpg")