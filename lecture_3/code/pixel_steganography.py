"""
Лекция 3: Стеганография — прячем текст в изображении.

Каждый 255-й пиксель хранит ASCII-код одной буквы секретного сообщения
в красном (R) канале. Визуально изображение почти не меняется.

Скрипт:
  1. Записывает секретный текст в photo_1.jpg → photo_1_secret.png
  2. Считывает текст обратно из photo_1_secret.png
"""

import numpy as np
from PIL import Image
from pathlib import Path

cwd = Path(__file__).parent.parent
IMAGE_PATH = cwd / "images/photo_1.jpg"
OUTPUT_PATH = cwd / "images/photo_1_secret.png"
SECRET_TEXT = "You got 5 for this colloquium"
STEP = 255  # каждый 255-й пиксель


# ──────────────────────────────────────────────
# 1. ЗАПИСЬ текста в изображение
# ──────────────────────────────────────────────
def hide_text(image_path: str, output_path: str, text: str, step: int = 255):
    """Прячет текст в изображении, записывая ASCII-код каждой буквы
    в R-канал каждого step-го пикселя."""

    img = np.array(Image.open(image_path).convert("RGB"))
    height, width, _ = img.shape
    total_pixels = height * width

    # Добавляем маркер конца строки (символ с кодом 0)
    encoded = text + "\x00"

    max_chars = total_pixels // step
    if len(encoded) > max_chars:
        raise ValueError(
            f"Текст слишком длинный! Максимум {max_chars - 1} символов, "
            f"а передано {len(text)}."
        )

    print(f"Изображение: {width}×{height} ({total_pixels} пикселей)")
    print(f"Шаг: каждый {step}-й пиксель")
    print(f"Максимум символов: {max_chars - 1}")
    print(f"Длина сообщения: {len(text)} символов")
    print()

    # Записываем каждую букву
    for i, char in enumerate(encoded):
        pixel_index = i * step
        y = pixel_index // width
        x = pixel_index % width

        old_r = img[y, x, 0]
        new_r = ord(char)  # ASCII-код символа

        img[y, x, 0] = new_r  # записываем в R-канал

        if char != "\x00":
            print(f"  Пиксель #{pixel_index:>6} ({y:>4}, {x:>4}): "
                  f"R: {old_r:>3} → {new_r:>3}  |  '{char}'")

    # Сохраняем как PNG (без потерь!), чтобы значения пикселей не изменились
    Image.fromarray(img).save(output_path)
    print(f"\n✅ Сохранено: {output_path}")


# ──────────────────────────────────────────────
# 2. ЧТЕНИЕ текста из изображения
# ──────────────────────────────────────────────
def read_text(image_path: str, step: int = 255) -> str:
    """Извлекает спрятанный текст из изображения."""

    img = np.array(Image.open(image_path))
    height, width, _ = img.shape
    total_pixels = height * width

    chars = []
    i = 0
    while True:
        pixel_index = i * step
        if pixel_index >= total_pixels:
            break

        y = pixel_index // width
        x = pixel_index % width

        code = img[y, x, 0]  # R-канал
        if code == 0:         # маркер конца
            break

        chars.append(chr(code))
        i += 1

    return "".join(chars)


# ──────────────────────────────────────────────
# ЗАПУСК
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("СТЕГАНОГРАФИЯ: запись текста в изображение")
    print("=" * 50)
    print(f"Секретный текст: \"{SECRET_TEXT}\"\n")

    hide_text(IMAGE_PATH, OUTPUT_PATH, SECRET_TEXT, STEP)

    print("\n" + "=" * 50)
    print("СТЕГАНОГРАФИЯ: чтение текста из изображения")
    print("=" * 50)

    result = read_text(OUTPUT_PATH, STEP)
    print(f"Прочитанный текст: \"{result}\"")

    if result == SECRET_TEXT:
        print("✅ Текст совпадает! Стеганография работает.")
    else:
        print("❌ Ошибка: тексты не совпадают.")
