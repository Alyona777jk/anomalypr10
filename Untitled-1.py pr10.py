# %%
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
        total_p = np.sum(self.p)
        if not np.isclose(total_p, 1.0, atol=0.01):
            raise ValueError(f"Помилка в {self.name}: Сума ймовірностей {total_p:.3f} != 1")

    def calculate_all(self, a_val, b_val):
        px = np.sum(self.p, axis=1)
        py = np.sum(self.p, axis=0)
        ex = np.sum(self.x * px)
        ey = np.sum(self.y * py)
        dx = np.sum((self.x**2) * px) - ex**2
        dy = np.sum((self.y**2) * py) - ey**2
        
        exy = 0
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                exy += self.x[i] * self.y[j] * self.p[i, j]
        corr = (exy - ex * ey) / np.sqrt(dx * dy) if dx * dy > 0 else 0
        
        y_idx = np.where(self.y == a_val)[0][0]
        p_y_a = py[y_idx]
        e_x_cond = np.sum(self.x * (self.p[:, y_idx] / p_y_a))
        x_idx = np.where(self.x == b_val)[0][0]
        p_x_b = px[x_idx]
        e_y_cond = np.sum(self.y * (self.p[x_idx, :] / p_x_b))
        f_stat = max(dx, dy) / min(dx, dy)
        p_f = 1 - stats.f.cdf(f_stat, len(self.x)-1, len(self.y)-1)

        return {
            "ex": ex, "ey": ey, "corr": corr,
            "e_x_cond": e_x_cond, "e_y_cond": e_y_cond,
            "f_stat": f_stat, "p_f": p_f
        }

def main():
    models = {
        "1": FinancialAnomalyModel("Варіант 10.1", [1, 4, 7, 10], [-2, 3, 8, 13], 
            [[0.02, 0.04, 0.03, 0.01], [0.06, 0.20, 0.10, 0.03], [0.06, 0.13, 0.11, 0.04], [0.03, 0.06, 0.06, 0.02]]),
        
        "2": FinancialAnomalyModel("Варіант 10.29", [-1, 1, 3, 5], [-5, -2, 1, 4], 
            [[0.01, 0.03, 0.03, 0.01], [0.03, 0.21, 0.10, 0.05], [0.04, 0.12, 0.13, 0.06], [0.02, 0.06, 0.07, 0.03]]),
        
        "3": FinancialAnomalyModel("Варіант 10.30", [-5, -2, 1, 4], [-6, -1, 4, 9], 
            [[0.02, 0.04, 0.04, 0.01], [0.05, 0.11, 0.10, 0.04], [0.06, 0.19, 0.12, 0.05], [0.03, 0.06, 0.06, 0.02]])
    }
    
    params = {"1": (3, 4), "2": (-2, 5), "3": (-1, -5)}

    while True:
        print(f"\n{'='*50}\nМЕНЮ МОДЕЛЮВАННЯ ФІНАНСОВИХ АНОМАЛІЙ\n{'='*50}")
        print("1. Розрахунок Варіанту 10.1  (a=3, b=4)")
        print("2. Розрахунок Варіанту 10.29 (a=-2, b=5)")
        print("3. Розрахунок Варіанту 10.30 (a=-1, b=-5)")
        print("4. Вихід")
        
        choice = input("\nОберіть номер завдання: ")
        
        if choice == "4": break
        if choice in models:
            m = models[choice]
            a, b = params[choice]
            res = m.calculate_all(a, b)
            
            print(f"\n>>> РЕЗУЛЬТАТИ ДЛЯ {m.name}:")
            print(f"Загальне E(X): {res['ex']:.3f} | E(Y): {res['ey']:.3f}")
            print(f"Коефіцієнт кореляції rxy: {res['corr']:.4f}")
            print(f"Умовне E(X) при Y={a}: {res['e_x_cond']:.3f}")
            print(f"Умовне E(Y) при X={b}: {res['e_y_cond']:.3f}")
            print("-" * 30)
            print(f"Критерій Фішера (F): {res['f_stat']:.3f}")
            print(f"Ймовірність адекватності (p-value): {res['p_f']:.4f}")
            status = "АДЕКВАТНА" if res['p_f'] > 0.05 else "АНОМАЛЬНА (Виявлено відхилення)"
            print(f"Висновок моделі: {status}")
        else:
            print("Помилка: Оберіть число від 1 до 4.")

if __name__ == "__main__":
    main()


