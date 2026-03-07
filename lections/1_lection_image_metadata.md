# Чтение и запись метаданных изображений форматов JPEG и PNG

---

## Содержание

1. [Введение в метаданные изображений](#1-введение-в-метаданные-изображений)
2. [Формат JPEG и его метаданные](#2-формат-jpeg-и-его-метаданные)
3. [Формат PNG и его метаданные](#3-формат-png-и-его-метаданные)
4. [Стандарты метаданных: EXIF, IPTC и XMP](#4-стандарты-метаданных-exif-iptc-и-xmp)
5. [Практика: чтение метаданных (Python)](#5-практика-чтение-метаданных-python)
6. [Практика: запись и изменение метаданных (Python)](#6-практика-запись-и-изменение-метаданных-python)
7. [Практика: работа с метаданными в Dart / Flutter](#7-практика-работа-с-метаданными-в-dart--flutter)
8. [Безопасность и конфиденциальность метаданных](#8-безопасность-и-конфиденциальность-метаданных)
9. [Контрольные вопросы](#9-контрольные-вопросы)
10. [Список литературы и ресурсов](#10-список-литературы-и-ресурсов)

---

## 1. Введение в метаданные изображений

**Метаданные** (metadata) — это «данные о данных». В контексте изображений метаданные содержат вспомогательную информацию, которая не является частью самого визуального содержимого, но описывает свойства и обстоятельства создания изображения.

### 1.1. Какую информацию хранят метаданные?

| Категория | Примеры данных |
|---|---|
| **Техническая** | Ширина и высота, глубина цвета, цветовое пространство, степень сжатия |
| **Камера и съёмка** | Модель камеры, выдержка, диафрагма (f-число), ISO, фокусное расстояние, вспышка |
| **Геолокация** | GPS-координаты (широта, долгота, высота) |
| **Авторские права** | Имя автора, лицензия, описание, ключевые слова |
| **Дата и время** | Дата съёмки, дата изменения, дата оцифровки |
| **Программная** | Название редактора, версия ПО, история редактирования |

### 1.2. Зачем нужны метаданные?

- **Организация и поиск** — каталогизация коллекций фотографий.
- **Авторское право** — встраивание информации о правообладателе.
- **Автоматическая обработка** — программы используют метаданные для автоматической ориентации снимков, привязки к карте и т.д.
- **Forensics (цифровая криминалистика)** — установление подлинности и обстоятельств создания изображения.

---

## 2. Формат JPEG и его метаданные

### 2.1. Общая структура файла JPEG

Файл JPEG состоит из последовательности **сегментов** (segments), каждый из которых начинается с двухбайтового **маркера**. Маркер всегда начинается с байта `0xFF`.

```
┌─────────────────────────────────────────────────────┐
│  SOI (Start Of Image)  │  0xFF 0xD8               │
├─────────────────────────────────────────────────────┤
│  APP0 (JFIF)           │  0xFF 0xE0  + данные     │
├─────────────────────────────────────────────────────┤
│  APP1 (EXIF)           │  0xFF 0xE1  + данные     │
├─────────────────────────────────────────────────────┤
│  APP1 (XMP)            │  0xFF 0xE1  + данные     │
├─────────────────────────────────────────────────────┤
│  APP13 (IPTC)          │  0xFF 0xED  + данные     │
├─────────────────────────────────────────────────────┤
│  DQT, DHT, SOF, SOS   │  таблицы квантования,    │
│                        │  Хаффмана, параметры     │
│                        │  кадра и начало скана    │
├─────────────────────────────────────────────────────┤
│  Сжатые данные изображения                         │
├─────────────────────────────────────────────────────┤
│  EOI (End Of Image)    │  0xFF 0xD9               │
└─────────────────────────────────────────────────────┘
```

### 2.2. Ключевые сегменты для метаданных

#### APP0 (JFIF) — `0xFF 0xE0`

Идентификатор формата JFIF (JPEG File Interchange Format). Содержит:
- Версию JFIF
- Единицы измерения плотности (DPI / пиксели на см)
- Встроенную миниатюру (thumbnail)

#### APP1 (EXIF) — `0xFF 0xE1`

Наиболее распространённый сегмент метаданных. Содержит структуры **EXIF** (Exchangeable Image File Format), организованные по принципу **TIFF-формата** (Tag-Image File Format):

```
APP1 Marker (0xFF 0xE1)
  ├── Длина сегмента (2 байта, big-endian)
  ├── "Exif\0\0" (6 байт — идентификатор)
  └── TIFF Header
       ├── Порядок байтов: "II" (little-endian) или "MM" (big-endian)
       ├── Магическое число: 0x002A
       ├── Смещение до первого IFD
       └── IFD0 (Image File Directory)
            ├── Тег 0x010F — Make (производитель камеры)
            ├── Тег 0x0110 — Model (модель камеры)
            ├── Тег 0x0112 — Orientation (ориентация)
            ├── Тег 0x8769 — Указатель на Exif Sub IFD
            │    ├── Тег 0x829A — ExposureTime (выдержка)
            │    ├── Тег 0x829D — FNumber (диафрагма)
            │    ├── Тег 0x8827 — ISOSpeedRatings
            │    ├── Тег 0x9003 — DateTimeOriginal
            │    └── ...
            ├── Тег 0x8825 — Указатель на GPS IFD
            │    ├── Тег 0x0001 — GPSLatitudeRef ("N" или "S")
            │    ├── Тег 0x0002 — GPSLatitude
            │    ├── Тег 0x0003 — GPSLongitudeRef ("E" или "W")
            │    ├── Тег 0x0004 — GPSLongitude
            │    └── ...
            └── IFD1 (миниатюра)
```

#### APP1 (XMP) — `0xFF 0xE1`

Содержит данные **XMP** (Extensible Metadata Platform) — стандарт Adobe, основанный на **RDF/XML**. Позволяет хранить произвольные метаданные в текстовом формате.

#### APP13 (IPTC) — `0xFF 0xED`

Содержит данные **IPTC-IIM** (International Press Telecommunications Council — Information Interchange Model). Используется в фотожурналистике для хранения заголовков, описаний, авторских прав и ключевых слов.

### 2.3. Формат хранения EXIF-тегов

Каждый тег EXIF представляет собой запись из **12 байтов** в IFD:

| Поле | Размер | Описание |
|---|---|---|
| Tag ID | 2 байта | Идентификатор тега (например, `0x010F` — Make) |
| Type | 2 байта | Тип данных (1=BYTE, 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL, ...) |
| Count | 4 байта | Количество значений |
| Value/Offset | 4 байта | Значение (если ≤ 4 байт) или смещение до данных |

**Пример:** тег `DateTimeOriginal` (0x9003) имеет тип ASCII и хранит строку формата `"2024:03:15 14:30:00"`.

---

## 3. Формат PNG и его метаданные

### 3.1. Общая структура файла PNG

Файл PNG состоит из **сигнатуры** (8 байт) и последовательности **чанков** (chunks):

```
┌─────────────────────────────────────────────────────┐
│  Сигнатура PNG: 89 50 4E 47 0D 0A 1A 0A           │
├─────────────────────────────────────────────────────┤
│  Чанк IHDR — заголовок изображения                 │
├─────────────────────────────────────────────────────┤
│  Чанк tEXt / iTXt / zTXt — текстовые метаданные   │
├─────────────────────────────────────────────────────┤
│  Чанк eXIf — EXIF-данные (с PNG 1.5+)             │
├─────────────────────────────────────────────────────┤
│  Чанк pHYs — физическое разрешение                 │
├─────────────────────────────────────────────────────┤
│  Чанк tIME — дата последнего изменения             │
├─────────────────────────────────────────────────────┤
│  Чанки IDAT — сжатые данные изображения            │
├─────────────────────────────────────────────────────┤
│  Чанк IEND — конец файла                           │
└─────────────────────────────────────────────────────┘
```

### 3.2. Структура чанка PNG

Каждый чанк имеет единообразную структуру:

| Поле | Размер | Описание |
|---|---|---|
| Length | 4 байта | Длина поля Data (big-endian) |
| Type | 4 байта | Тип чанка (ASCII, например `tEXt`) |
| Data | Length байт | Собственно данные чанка |
| CRC | 4 байта | Контрольная сумма CRC-32 полей Type + Data |

> **Соглашение об именовании чанков:**
> - Первая буква строчная → чанк **вспомогательный** (ancillary), может быть игнорирован.
> - Первая буква заглавная → чанк **критический** (critical), обязателен для отображения.

### 3.3. Текстовые метаданные PNG

PNG определяет три типа текстовых чанков:

#### tEXt — простой текст (Latin-1)

```
┌──────────────────────────────────────────┐
│ Keyword (1–79 байт, Latin-1)            │
│ Разделитель: 0x00 (null byte)           │
│ Text (произвольная длина, Latin-1)       │
└──────────────────────────────────────────┘
```

#### zTXt — сжатый текст (zlib + Latin-1)

```
┌──────────────────────────────────────────┐
│ Keyword (1–79 байт, Latin-1)            │
│ Null-разделитель: 0x00                  │
│ Compression method: 0x00 (zlib deflate) │
│ Compressed text (сжатый Latin-1)        │
└──────────────────────────────────────────┘
```

#### iTXt — интернационализированный текст (UTF-8)

```
┌──────────────────────────────────────────┐
│ Keyword (1–79 байт, Latin-1)            │
│ Null-разделитель: 0x00                  │
│ Compression flag: 0 или 1              │
│ Compression method: 0x00               │
│ Language tag (например, "ru")           │
│ Null-разделитель: 0x00                  │
│ Translated keyword (UTF-8)             │
│ Null-разделитель: 0x00                  │
│ Text (UTF-8, опционально сжатый)       │
└──────────────────────────────────────────┘
```

**Стандартные ключевые слова PNG:**

| Ключевое слово | Описание |
|---|---|
| `Title` | Краткое название изображения |
| `Author` | Имя автора |
| `Description` | Описание содержимого |
| `Copyright` | Уведомление об авторских правах |
| `Creation Time` | Дата создания |
| `Software` | ПО, создавшее изображение |
| `Comment` | Произвольный комментарий |
| `Source` | Устройство, на котором создано изображение |

### 3.4. Чанк eXIf (PNG 1.5)

Начиная с расширения спецификации PNG в 2017 году, появился официальный чанк `eXIf`, который содержит **EXIF-данные в формате TIFF**, аналогично сегменту APP1 в JPEG, но **без** заголовка `"Exif\0\0"`.

### 3.5. Чанк pHYs — физическое разрешение

```
┌──────────────────────────────────────────┐
│ Pixels per unit, X axis (4 байта)       │
│ Pixels per unit, Y axis (4 байта)       │
│ Unit specifier: 0=unknown, 1=метр       │
└──────────────────────────────────────────┘
```

### 3.6. Чанк tIME — время модификации

```
┌──────────────────────────────────────────┐
│ Year   (2 байта, big-endian)            │
│ Month  (1 байт, 1–12)                  │
│ Day    (1 байт, 1–31)                  │
│ Hour   (1 байт, 0–23)                  │
│ Minute (1 байт, 0–59)                  │
│ Second (1 байт, 0–60, 60 для leap sec) │
└──────────────────────────────────────────┘
```

---

## 4. Стандарты метаданных: EXIF, IPTC и XMP

### 4.1. Сравнительная таблица

| Характеристика | EXIF | IPTC-IIM | XMP |
|---|---|---|---|
| **Разработчик** | JEITA / CIPA | IPTC | Adobe |
| **Формат данных** | Бинарный (TIFF) | Бинарный | XML (текстовый) |
| **Кодировка текста** | ASCII / JIS | ASCII / UTF-8 | UTF-8 |
| **Расширяемость** | Ограничена | Ограничена | Высокая (пространства имён) |
| **Поддержка в JPEG** | APP1 | APP13 | APP1 |
| **Поддержка в PNG** | eXIf чанк | Не стандартно | iTXt чанк |
| **Типичное применение** | Параметры камеры, GPS | Фотожурналистика | Универсальное |

### 4.2. Подробнее о XMP

XMP использует модель данных **RDF** (Resource Description Framework) и записывается в виде XML:

```xml
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:exif="http://ns.adobe.com/exif/1.0/">
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Закат над горами</rdf:li>
        </rdf:Alt>
      </dc:title>
      <dc:creator>
        <rdf:Seq>
          <rdf:li>Иван Петров</rdf:li>
        </rdf:Seq>
      </dc:creator>
      <xmp:CreateDate>2024-03-15T14:30:00+06:00</xmp:CreateDate>
      <exif:FocalLength>50/1</exif:FocalLength>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
```

**Важно:** отступы `<?xpacket ... ?>` создают «пакет» фиксированного размера, позволяя редактировать XMP на месте без перезаписи всего файла.

---

## 5. Практика: чтение метаданных (Python)

### 5.1. Установка библиотек

```bash
pip install Pillow piexif
```

### 5.2. Чтение EXIF из JPEG с помощью Pillow

```python
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
meta = read_exif("photo.jpg")
for key, value in meta.items():
    print(f"{key}: {value}")
```

**Вывод (пример):**

```
Make: Canon
Model: Canon EOS R5
DateTime: 2024:03:15 14:30:00
ExposureTime: (1, 250)
FNumber: (28, 10)
ISOSpeedRatings: 400
FocalLength: (50, 1)
...
```

### 5.3. Извлечение GPS-координат

```python
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_gps_info(filepath: str) -> dict | None:
    """Извлекает GPS-координаты из EXIF."""
    image = Image.open(filepath)
    exif_data = image._getexif()

    if exif_data is None:
        return None

    gps_info = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id)
        if tag_name == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_info[gps_tag_name] = gps_value
            return gps_info

    return None


def dms_to_decimal(dms_tuple, ref: str) -> float:
    """Конвертирует координаты из DMS в десятичный формат.

    Args:
        dms_tuple: Кортеж (градусы, минуты, секунды)
        ref: Ссылка направления ('N', 'S', 'E', 'W')
    """
    degrees = float(dms_tuple[0])
    minutes = float(dms_tuple[1])
    seconds = float(dms_tuple[2])

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if ref in ('S', 'W'):
        decimal = -decimal

    return decimal


# Использование:
gps = get_gps_info("photo.jpg")
if gps:
    lat = dms_to_decimal(gps['GPSLatitude'], gps['GPSLatitudeRef'])
    lon = dms_to_decimal(gps['GPSLongitude'], gps['GPSLongitudeRef'])
    print(f"Широта:  {lat:.6f}")
    print(f"Долгота: {lon:.6f}")
    print(f"Google Maps: https://maps.google.com/?q={lat},{lon}")
```

### 5.4. Чтение текстовых метаданных PNG

```python
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
```

### 5.5. Чтение метаданных с помощью piexif

Библиотека `piexif` даёт доступ к «сырым» EXIF-данным:

```python
import piexif

def read_raw_exif(filepath: str):
    """Читает все IFD из EXIF с использованием piexif."""
    exif_dict = piexif.load(filepath)

    ifd_names = {
        "0th":   piexif.ImageIFD,     # IFD0 — основное изображение
        "Exif":  piexif.ExifIFD,      # Exif Sub IFD
        "GPS":   piexif.GPSIFD,       # GPS IFD
        "1st":   piexif.ImageIFD,     # IFD1 — миниатюра
    }

    for ifd_name in ("0th", "Exif", "GPS", "1st"):
        print(f"\n=== {ifd_name} IFD ===")
        for tag_id, value in exif_dict[ifd_name].items():
            tag_name = piexif.TAGS[ifd_name].get(tag_id, {}).get("name", f"Tag_{tag_id}")
            print(f"  {tag_name} (0x{tag_id:04X}): {value}")

    # Миниатюра
    if exif_dict.get("thumbnail"):
        print(f"\nМиниатюра: {len(exif_dict['thumbnail'])} байт")


# Использование:
read_raw_exif("photo.jpg")
```

---

## 6. Практика: запись и изменение метаданных (Python)

### 6.1. Запись EXIF в JPEG с помощью piexif

```python
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
    exif_dict["0th"][piexif.ImageIFD.Make] = "MyCamera"
    exif_dict["0th"][piexif.ImageIFD.Model] = "Model X100"
    exif_dict["0th"][piexif.ImageIFD.Software] = "Python piexif"
    exif_dict["0th"][piexif.ImageIFD.Artist] = "Иван Петров".encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Copyright] = "© 2024 Иван Петров".encode("utf-8")

    # Записываем данные в Exif Sub IFD
    now = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = now
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = now
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
        "Тестовое фото для лекции", encoding="unicode"
    )

    # Записываем GPS-данные (Бишкек: 42.8746°N, 74.5698°E)
    lat_deg = _decimal_to_dms(42.8746)
    lon_deg = _decimal_to_dms(74.5698)

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
write_exif_to_jpeg("input.jpg", "output_with_exif.jpg")
```

### 6.2. Удаление всех EXIF-данных

```python
import piexif

def strip_exif(filepath: str, output_path: str):
    """Полностью удаляет EXIF-данные из JPEG-файла."""
    piexif.remove(filepath, new_file=output_path)
    print(f"EXIF удалены. Результат: {output_path}")


# Использование:
strip_exif("photo_with_gps.jpg", "photo_clean.jpg")
```

### 6.3. Запись текстовых метаданных в PNG

```python
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
```

### 6.4. Копирование EXIF между файлами

```python
import piexif

def copy_exif(source_path: str, target_path: str, output_path: str):
    """Копирует EXIF из одного JPEG в другой."""
    # Извлекаем EXIF из исходного файла
    exif_dict = piexif.load(source_path)
    exif_bytes = piexif.dump(exif_dict)

    # Вставляем в целевой файл
    piexif.insert(exif_bytes, target_path, new_file=output_path)
    print(f"EXIF скопированы из {source_path} в {output_path}")


# Использование:
copy_exif("original.jpg", "edited.jpg", "edited_with_exif.jpg")
```

---

## 7. Практика: работа с метаданными в Dart / Flutter

### 7.1. Пакеты для работы с метаданными

| Пакет | Описание |
|---|---|
| `exif` | Чтение EXIF-данных из JPEG и TIFF |
| `image` | Полная библиотека для работы с изображениями, включая метаданные |
| `native_exif` | Нативный доступ к EXIF через платформенные API (iOS/Android) |

### 7.2. Чтение EXIF в Dart с пакетом `exif`

```yaml
# pubspec.yaml
dependencies:
  exif: ^3.3.0
```

```dart
import 'dart:io';
import 'package:exif/exif.dart';

Future<void> readExifFromFile(String filePath) async {
  final fileBytes = await File(filePath).readAsBytes();
  final tags = await readExifFromBytes(fileBytes);

  if (tags.isEmpty) {
    print('EXIF-данные не найдены.');
    return;
  }

  // Вывод всех тегов
  for (final entry in tags.entries) {
    print('${entry.key}: ${entry.value}');
  }

  // Извлечение конкретных значений
  final make = tags['Image Make']?.printable;
  final model = tags['Image Model']?.printable;
  final dateTime = tags['EXIF DateTimeOriginal']?.printable;
  final iso = tags['EXIF ISOSpeedRatings']?.printable;

  print('\n--- Основные данные ---');
  print('Камера: $make $model');
  print('Дата съёмки: $dateTime');
  print('ISO: $iso');

  // Извлечение GPS
  final latRef = tags['GPS GPSLatitudeRef']?.printable;
  final lat = tags['GPS GPSLatitude']?.printable;
  final lonRef = tags['GPS GPSLongitudeRef']?.printable;
  final lon = tags['GPS GPSLongitude']?.printable;

  if (lat != null && lon != null) {
    print('GPS: $lat $latRef, $lon $lonRef');
  }
}
```

### 7.3. Использование `native_exif` для чтения и записи

```yaml
# pubspec.yaml
dependencies:
  native_exif: ^0.6.0
```

```dart
import 'package:native_exif/native_exif.dart';

Future<void> readAndWriteExif(String filePath) async {
  // Открываем файл для работы с EXIF
  final exif = await Exif.fromPath(filePath);

  // Чтение атрибутов
  final attributes = await exif.getAttributes();
  print('Все атрибуты:');
  attributes?.forEach((key, value) {
    print('  $key: $value');
  });

  // Чтение конкретных полей
  final dateTime = await exif.getOriginalDate();
  final latLong = await exif.getLatLong();

  print('Дата съёмки: $dateTime');
  if (latLong != null) {
    print('Координаты: ${latLong.latitude}, ${latLong.longitude}');
  }

  // Запись атрибутов
  await exif.writeAttribute('UserComment', 'Комментарий из Flutter');
  await exif.writeAttributes({
    'Artist': 'Иван Петров',
    'Copyright': '© 2024',
  });

  // Закрываем (сохраняем изменения)
  await exif.close();
  print('EXIF-данные обновлены.');
}
```

### 7.4. Ручной разбор PNG-метаданных в Dart

```dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Читает текстовые чанки (tEXt) из PNG-файла.
Future<Map<String, String>> readPngTextChunks(String filePath) async {
  final bytes = await File(filePath).readAsBytes();
  final metadata = <String, String>{};

  // Проверяем PNG-сигнатуру
  final pngSignature = [137, 80, 78, 71, 13, 10, 26, 10];
  for (int i = 0; i < 8; i++) {
    if (bytes[i] != pngSignature[i]) {
      throw FormatException('Файл не является PNG.');
    }
  }

  // Проходим по чанкам
  int offset = 8; // после сигнатуры

  while (offset < bytes.length) {
    // Длина данных (4 байта, big-endian)
    final dataLength = _readUint32BE(bytes, offset);
    offset += 4;

    // Тип чанка (4 байта ASCII)
    final chunkType = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    offset += 4;

    // Данные чанка
    final chunkData = bytes.sublist(offset, offset + dataLength);
    offset += dataLength;

    // CRC (4 байта) — пропускаем
    offset += 4;

    // Обрабатываем текстовые чанки
    if (chunkType == 'tEXt') {
      final nullIndex = chunkData.indexOf(0);
      if (nullIndex != -1) {
        final key = latin1.decode(chunkData.sublist(0, nullIndex));
        final value = latin1.decode(chunkData.sublist(nullIndex + 1));
        metadata[key] = value;
      }
    } else if (chunkType == 'iTXt') {
      final nullIndex = chunkData.indexOf(0);
      if (nullIndex != -1) {
        final key = latin1.decode(chunkData.sublist(0, nullIndex));
        // После ключа: null, compression flag, compression method,
        // language tag (null-terminated), translated keyword (null-terminated),
        // text
        int pos = nullIndex + 1;
        final compressionFlag = chunkData[pos++];
        pos++; // compression method

        // Language tag
        final langEnd = chunkData.indexOf(0, pos);
        pos = langEnd + 1;

        // Translated keyword
        final tkeyEnd = chunkData.indexOf(0, pos);
        pos = tkeyEnd + 1;

        // Text
        if (compressionFlag == 0) {
          final value = utf8.decode(chunkData.sublist(pos));
          metadata[key] = value;
        }
        // При compressionFlag == 1 нужна распаковка zlib
      }
    }

    // Завершаем при IEND
    if (chunkType == 'IEND') break;
  }

  return metadata;
}

int _readUint32BE(Uint8List bytes, int offset) {
  return (bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3];
}
```

---

## 8. Безопасность и конфиденциальность метаданных

### 8.1. Угрозы конфиденциальности

| Угроза | Описание | Пример |
|---|---|---|
| **Геолокация** | GPS-координаты раскрывают местоположение | Фото дома → адрес проживания |
| **Идентификация устройства** | Серийный номер камеры/телефона | Связывание анонимных фото с владельцем |
| **Временные метки** | Точное время создания | Установление распорядка дня |
| **Миниатюры** | EXIF-миниатюра может содержать обрезанный оригинал | Кадрированное фото сохраняет исходный thumbnail |

### 8.2. Рекомендации по защите

1. **Очищайте метаданные** перед публикацией в интернете.
2. **Отключайте запись GPS** в настройках камеры/телефона, если геолокация не нужна.
3. **Проверяйте миниатюры** — они могут содержать информацию, удалённую из основного изображения.
4. **Используйте инструменты** для массовой очистки:
   - `ExifTool` (командная строка) — мощный универсальный инструмент
   - `MAT2` (Metadata Anonymisation Toolkit) — для анонимизации
   - Встроенные функции ОС (Windows: Свойства → Подробности → Удалить)

### 8.3. Инструмент ExifTool — основные команды

```bash
# Просмотреть все метаданные
exiftool photo.jpg

# Просмотреть только GPS
exiftool -gps:all photo.jpg

# Удалить все метаданные
exiftool -all= photo.jpg

# Удалить только GPS
exiftool -gps:all= photo.jpg

# Записать автора
exiftool -Artist="Иван Петров" photo.jpg

# Скопировать метаданные между файлами
exiftool -TagsFromFile source.jpg target.jpg

# Массовая обработка всех JPEG в папке
exiftool -all= -overwrite_original ./photos/*.jpg
```

---

## 9. Контрольные вопросы

1. Что такое метаданные изображения и какие категории информации они могут содержать?

2. Какой маркер в JPEG-файле обозначает сегмент с EXIF-данными? Какова внутренняя структура этого сегмента?

3. В чём различие между текстовыми чанками `tEXt`, `zTXt` и `iTXt` в формате PNG?

4. Чем отличаются стандарты EXIF, IPTC и XMP? Для каких задач лучше подходит каждый из них?

5. Опишите структуру записи тега в IFD (Image File Directory). Какие поля она содержит?

6. Каким образом GPS-координаты хранятся в EXIF? Как конвертировать представление DMS (градусы-минуты-секунды) в десятичный формат?

7. Почему метаданные могут представлять угрозу для конфиденциальности? Приведите конкретные примеры.

8. Напишите функцию на Python, которая:
   - Открывает JPEG-файл
   - Извлекает дату съёмки и модель камеры
   - Выводит результат в консоль

9. Напишите функцию на Python, которая записывает текстовые метаданные (Title, Author, Description) в PNG-файл с поддержкой кириллицы (UTF-8).

10. Каким образом можно хранить EXIF-данные в файле PNG? Какой чанк для этого предназначен?

---

## 10. Список литературы и ресурсов

1. **Спецификация EXIF 2.32** — [https://www.cipa.jp/std/documents/e/DC-008-Translation-2019-E.pdf](https://www.cipa.jp/std/documents/e/DC-008-Translation-2019-E.pdf)

2. **Спецификация PNG** — [https://www.w3.org/TR/png/](https://www.w3.org/TR/png/)

3. **Стандарт XMP** — [https://www.adobe.com/devnet/xmp.html](https://www.adobe.com/devnet/xmp.html)

4. **Стандарт IPTC Photo Metadata** — [https://iptc.org/standards/photo-metadata/](https://iptc.org/standards/photo-metadata/)

5. **Спецификация JPEG (ITU-T T.81)** — [https://www.w3.org/Graphics/JPEG/itu-t81.pdf](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)

6. **ExifTool** — [https://exiftool.org/](https://exiftool.org/)

7. **Pillow (Python Imaging Library)** — [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)

8. **piexif** — [https://pypi.org/project/piexif/](https://pypi.org/project/piexif/)

9. **Пакет exif для Dart** — [https://pub.dev/packages/exif](https://pub.dev/packages/exif)

10. **Пакет native_exif для Flutter** — [https://pub.dev/packages/native_exif](https://pub.dev/packages/native_exif)

---

*Лекция подготовлена: Февраль 2026*
