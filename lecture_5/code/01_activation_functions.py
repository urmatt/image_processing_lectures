"""
Лекция 5 — Пример 1: Визуализация функций активации нейронных сетей.
Демонстрирует Sigmoid, ReLU и Tanh — основные функции активации.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

# Функции активации
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

functions = [
    (sigmoid, 'Sigmoid: σ(z) = 1/(1+e⁻ᶻ)', 'tab:blue'),
    (relu, 'ReLU: f(z) = max(0, z)', 'tab:orange'),
    (tanh, 'Tanh: tanh(z)', 'tab:green'),
]

for ax, (y, name, color) in zip(axes, functions):
    ax.plot(x, y, linewidth=2.5, color=color)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('z')
    ax.set_ylabel('f(z)')

plt.suptitle("Функции активации нейронных сетей", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150, bbox_inches='tight')
plt.show()
print("Сохранено: activation_functions.png")
