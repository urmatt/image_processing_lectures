import numpy as np
import matplotlib.pyplot as plt

# dtype=np.uint8 обязателен для изображений (числа от 0 до 255)
img_gray = np.zeros((4, 4), dtype=np.uint8)

print("Исходная матрица:\n", img_gray)

# 2. Заполняем пиксели вручную
# Координаты: [строка (y), столбец (x)]

# Левый верхний угол - белый
# img_gray[0, 0] = 255

img_gray[:, :] = 255

# Правый нижний угол - серый
img_gray[3, 3] = 128
img_gray[0, 0] = 0

# Заполним вторую строку (индекс 1) светло-серым
# img_gray[1, :] = 200

# Заполним третий столбец (индекс 2) темно-серым
# img_gray[:, 2] = 50

print("\nЗаполненная матрица:\n", img_gray)

# 3. Визуализация
plt.figure(figsize=(4, 4))
# Важно: cmap='gray' для ЧБ, interpolation='nearest' чтобы не размывало пиксели
plt.imshow(img_gray, cmap='gray', interpolation='nearest')
plt.title("Ч/Б изображение 4x4")
plt.axis('off') # Убираем оси координат
plt.show()