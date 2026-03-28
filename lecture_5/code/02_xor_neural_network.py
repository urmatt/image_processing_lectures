"""
Лекция 5 — Пример 2: Простейшая нейронная сеть для задачи XOR (без библиотек).
Демонстрирует обучение 2-слойной сети с обратным распространением ошибки.
"""
import numpy as np

# ============================================================
# Данные: задача XOR (не решается одним нейроном!)
# ============================================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ============================================================
# Инициализация весов
# ============================================================
np.random.seed(42)
weights_hidden = np.random.randn(2, 4) * 0.5   # 2 входа → 4 скрытых нейрона
bias_hidden = np.zeros((1, 4))
weights_output = np.random.randn(4, 1) * 0.5   # 4 скрытых → 1 выход
bias_output = np.zeros((1, 1))


def sigmoid(z):
    """Функция активации Sigmoid."""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Производная Sigmoid (вход — уже результат sigmoid!)."""
    return z * (1 - z)


# ============================================================
# Обучение
# ============================================================
learning_rate = 1.0
epochs = 10000
losses = []

for epoch in range(epochs):
    # --- Прямой проход (Forward Pass) ---
    hidden_input = X @ weights_hidden + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ weights_output + bias_output
    final_output = sigmoid(final_input)

    # --- Вычисление ошибки ---
    error = y - final_output
    loss = np.mean(error ** 2)
    losses.append(loss)

    # --- Обратное распространение (Backpropagation) ---
    # Градиент выходного слоя
    d_output = error * sigmoid_derivative(final_output)

    # Градиент скрытого слоя
    error_hidden = d_output @ weights_output.T
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # --- Обновление весов ---
    weights_output += hidden_output.T @ d_output * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += X.T @ d_hidden * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 2000 == 0:
        print(f"Эпоха {epoch:5d} / {epochs}, loss = {loss:.6f}")

# ============================================================
# Результат
# ============================================================
print("\n" + "=" * 40)
print("Предсказания после обучения:")
print("=" * 40)
for i in range(4):
    predicted = final_output[i][0]
    expected = y[i][0]
    status = "✓" if abs(predicted - expected) < 0.1 else "✗"
    print(f"  Вход: {X[i]}  →  Предсказание: {predicted:.4f}  "
          f"(Ожидали: {expected})  {status}")

# Визуализация процесса обучения
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=1.5, color='tab:blue')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (MSE)")
    plt.title("Обучение нейронной сети (задача XOR)")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig("xor_training.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nГрафик сохранён: xor_training.png")
except ImportError:
    print("\n(matplotlib не установлен — график не построен)")
