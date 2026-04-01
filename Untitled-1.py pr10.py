import numpy as np
import scipy.stats as stats


class FinancialAnomalyModel:
    def __init__(self, name, x_values, y_values, matrix):
        self.name = name
        self.x = np.array(x_values)
        self.y = np.array(y_values)
        self.p = np.array(matrix)

        self.validate_data()
    def validate_data(self):
        if self.p.shape != (len(self.x), len(self.y)):
            raise ValueError(f"{self.name}: Розмірність матриці не відповідає X і Y")

        total_p = np.sum(self.p)
        if not np.isclose(total_p, 1.0, atol=0.01):
            raise ValueError(f"{self.name}: Сума ймовірностей = {total_p:.3f}, повинна бути ≈1")
    def safe_index(self, arr, value, var_name):
        idx = np.where(arr == value)[0]
        if len(idx) == 0:
            raise ValueError(f"{self.name}: Значення {var_name}={value} відсутнє")
        return idx[0]
    def calculate_all(self, a_val, b_val):
        px = np.sum(self.p, axis=1)
        py = np.sum(self.p, axis=0)
        ex = np.sum(self.x * px)
        ey = np.sum(self.y * py)
        dx = np.sum((self.x ** 2) * px) - ex ** 2
        dy = np.sum((self.y ** 2) * py) - ey ** 2

        if dx <= 0 or dy <= 0:
            raise ValueError(f"{self.name}: Некоректна дисперсія")

        exy = sum(self.x[i] * self.y[j] * self.p[i, j]
                  for i in range(len(self.x))
                  for j in range(len(self.y)))

        corr = (exy - ex * ey) / np.sqrt(dx * dy)
        y_idx = self.safe_index(self.y, a_val, "Y")
        p_y_a = py[y_idx]

        if p_y_a == 0:
            raise ValueError(f"{self.name}: P(Y={a_val}) = 0")

        e_x_cond = np.sum(self.x * (self.p[:, y_idx] / p_y_a))

        x_idx = self.safe_index(self.x, b_val, "X")
        p_x_b = px[x_idx]

        if p_x_b == 0:
            raise ValueError(f"{self.name}: P(X={b_val}) = 0")

        e_y_cond = np.sum(self.y * (self.p[x_idx, :] / p_x_b))
        f_stat = max(dx, dy) / min(dx, dy)
        p_value = 1 - stats.f.cdf(f_stat, len(self.x) - 1, len(self.y) - 1)
        is_adequate = p_value > 0.05

        return {
            "ex": ex,
            "ey": ey,
            "dx": dx,
            "dy": dy,
            "corr": corr,
            "e_x_cond": e_x_cond,
            "e_y_cond": e_y_cond,
            "f_stat": f_stat,
            "p_value": p_value,
            "adequate": is_adequate
        }


def main():
    try:
        models = {
            "1": FinancialAnomalyModel(
                "Варіант 10.1",
                [1, 4, 7, 10],
                [-2, 3, 8, 13],
                [[0.02, 0.04, 0.03, 0.01],
                 [0.06, 0.20, 0.10, 0.03],
                 [0.06, 0.13, 0.11, 0.04],
                 [0.03, 0.06, 0.06, 0.02]]
            ),
            "2": FinancialAnomalyModel(
                "Варіант 10.29",
                [-1, 1, 3, 5],
                [-5, -2, 1, 4],
                [[0.01, 0.03, 0.03, 0.01],
                 [0.03, 0.21, 0.10, 0.05],
                 [0.04, 0.12, 0.13, 0.06],
                 [0.02, 0.06, 0.07, 0.03]]
            ),
            "3": FinancialAnomalyModel(
                "Варіант 10.30",
                [-5, -2, 1, 4],
                [-6, -1, 4, 9],
                [[0.02, 0.04, 0.04, 0.01],
                 [0.05, 0.11, 0.10, 0.04],
                 [0.06, 0.19, 0.12, 0.05],
                 [0.03, 0.06, 0.06, 0.02]]
            )
        }
        params = {
            "1": (3, 4),
            "2": (-2, 5),
            "3": (-1, 4) 
        }

    except Exception as e:
        print(f"Помилка ініціалізації моделей: {e}")
        return

    while True:
        print("\n" + "=" * 50)
        print("МЕНЮ МОДЕЛЮВАННЯ ФІНАНСОВИХ АНОМАЛІЙ")
        print("=" * 50)
        print("1 - Варіант 10.1")
        print("2 - Варіант 10.29")
        print("3 - Варіант 10.30")
        print("0 - Вихід")

        choice = input("Оберіть пункт: ")

        if choice == "0":
            print("Завершення роботи програми.")
            break

        if choice not in models:
            print(" Невірний вибір!")
            continue

        try:
            model = models[choice]
            a, b = params[choice]

            result = model.calculate_all(a, b)

            print(f"\n РЕЗУЛЬТАТИ: {model.name}")
            print(f"E(X) = {result['ex']:.3f}")
            print(f"E(Y) = {result['ey']:.3f}")
            print(f"D(X) = {result['dx']:.3f}")
            print(f"D(Y) = {result['dy']:.3f}")
            print(f"rxy = {result['corr']:.4f}")

            print(f"\nУмовні сподівання:")
            print(f"E(X | Y={a}) = {result['e_x_cond']:.3f}")
            print(f"E(Y | X={b}) = {result['e_y_cond']:.3f}")

            print("\nКритерій Фішера:")
            print(f"F = {result['f_stat']:.3f}")
            print(f"p-value = {result['p_value']:.4f}")

            if result['adequate']:
                print(" Модель АДЕКВАТНА")
            else:
                print("⚠️ Модель АНомальна (є відхилення)")

        except Exception as e:
            print(f" Помилка обчислення: {e}")


if __name__ == "__main__":
    main()
