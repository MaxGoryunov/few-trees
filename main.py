import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import train_test_split
import math


def generate_spiral_points(revolutions, num_points):
    t = np.linspace(0, 2 * np.pi * revolutions, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return x, y, t


def classify_points(t):
    labels = np.where(t < 2 * math.pi, 'A', 'B')
    return labels


def main():
    revolutions = 3
    num_points = 300

    x, y, t = generate_spiral_points(revolutions, num_points)
    labels = classify_points(t)

    X = np.array(list(zip(x, y, t)))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model = RandomTreesEmbedding(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.transform(X_test)
    print(predictions)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=['blue' if label == 'A' else 'red' for label in labels], alpha=0.5, label='Data points')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='x', label='Test points')
    plt.title('Spiral Points Classification')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
