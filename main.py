import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def create_features(X):
    # Приводим к float64 (если данные не в этом формате)
    X = np.asarray(X, dtype=np.float64)

    # Заменяем NaN и бесконечные значения на 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Ограничиваем значения, чтобы избежать проблем с exp/log
    X_clipped = np.clip(X, -100, 100)

    # Генерируем новые признаки
    X_squared = X_clipped ** 2
    X_cubed = X_clipped ** 3
    X_log = np.log(np.abs(X_clipped) + 1e-5)  # Защита от log(0)
    X_sqrt = np.sqrt(np.abs(X_clipped))
    X_exp = np.exp(np.clip(X_clipped, -20, 20))  # Защита от переполнения exp

    # Объединяем все признаки
    X_combined = np.hstack((X_clipped, X_squared, X_cubed, X_log, X_sqrt, X_exp))

    # Еще раз очищаем результат от возможных NaN или inf
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    return X_combined

# Применяем преобразование


def wolf_g(alpha, x, y):
    return -(np.sum(alpha) - 1 / 2 * alpha.T @ (np.diag(y) @ (x @ x.T) @ np.diag(y)) @ alpha)


def predict_g(x, w, b):
    return np.sign(x @ w + b)


def generate_ring_data(n_center, n_ring, radius=0.8, ring_noise=0.05):
    rng = np.random.default_rng(42)
    X_center = rng.normal(scale=0.2, size=(n_center, 2))
    angles = rng.uniform(0, 2 * np.pi, size=n_ring)
    radii = rng.normal(loc=radius, scale=ring_noise, size=n_ring)
    X_ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([X_center, X_ring])


def k(sig, x1, x2):
    x1_sq = np.sum(x1 ** 2, axis=1).reshape(-1, 1)
    x2_sq = np.sum(x2 ** 2, axis=1).reshape(1, -1)
    dists = x1_sq + x2_sq - 2 * np.dot(x1, x2.T)
    return np.exp(-dists / (2 * sig ** 2))


def wolf_b(alpha, x, y, sig):
    return -(np.sum(alpha) - 1 / 2 * alpha.T @ (np.diag(y) @ k(sig, x, x) @ np.diag(y)) @ alpha)


def predict_b(x, x_sv, y_sv, alpha, b, sig):
    return np.sign((alpha * y_sv) @ k(sig, x_sv, x) + b)


def plot_svm_nonlinear_decision(x_sv, y_sv, alpha_sv, b, sig, bounds=None, resolution=300):
    def decision_function(x_input):
        return (alpha_sv * y_sv) @ k(sig, x_sv, x_input) + b

    if bounds is None:
        x_min, x_max = x_sv[:, 0].min() - 1, x_sv[:, 0].max() + 1
        y_min, y_max = x_sv[:, 1].min() - 1, x_sv[:, 1].max() + 1
    else:
        (x_min, x_max), (y_min, y_max) = bounds

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = decision_function(grid).reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[-1, 0, 1], colors=['black', 'black', 'black'], linestyles=['--', '-', '--'])


def compute_margin(alpha, y, x, sig):
    K = k(sig, x, x)
    margin_sq = np.sum((alpha * y).reshape(-1, 1) * (alpha * y).reshape(1, -1) * K)
    return 2 / np.sqrt(margin_sq)


def cross_val_score_svm(x, y, c, sig, m=5):
    kf = KFold(n_splits=m, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        n_train = len(y_train)
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_train)}
        bounds = [(0, c)] * n_train
        alpha0 = np.zeros(n_train)

        result = minimize(fun=wolf_b, x0=alpha0, bounds=bounds, constraints=constraints, args=(x_train, y_train, sig,))
        alpha = result.x
        sv_i = (alpha > tol) * (alpha < c - tol)

        if np.sum(sv_i) == 0:
            scores.append(0)
            continue

        b = np.mean(y_train[sv_i] - (alpha[sv_i] * y_train[sv_i]) @ k(sig, x_train[sv_i], x_train[sv_i]))
        pred = predict_b(x_val, x_train[sv_i], y_train[sv_i], alpha[sv_i], b, sig)
        acc = np.mean(pred == y_val)
        scores.append(acc)

    return np.mean(scores)


n = 200
tol = 0.00001
sig = np.sqrt(1)
c = 10

rng = np.random.default_rng(52)
x1 = rng.normal(scale=0.2, size=(n // 2, 2)) + 1
x2 = rng.normal(scale=0.2, size=(n // 2, 2)) + 2.5
x = np.vstack([x1, x2])
y = []
y.extend([-1] * (n // 2))
y.extend([1] * (n // 2))
X_extended = create_features(x)
x_train, x_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=47)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train[11] *= -1
y_train[49] *= -1
y_train[96] *= -1
#
n_train = len(y_train)
constraints = {
    'type': 'eq',
    'fun': lambda alpha: np.dot(alpha, y_train),
}
bounds = [(0, c) for _ in range(n_train)]
alpha0 = np.zeros(n_train)
result = minimize(fun=wolf_g, x0=alpha0, bounds=bounds, constraints=constraints, args=(x_train, y_train,))
alpha = result.x
sv_i = (alpha > tol) * (alpha < c - tol)
bsv_i = (alpha >= c - tol) * (alpha <= c + tol)
w = alpha[sv_i] @ np.diag(y_train[sv_i]) @ x_train[sv_i]
b = np.mean(y_train[sv_i] - x_train[sv_i] @ w)
pred = predict_g(x_train, w, b)
M = 2 / np.linalg.norm(w)
print(f'Число опорных векторов: {sum(sv_i)}')
print(f'Число связанных опорных векторов: {sum(bsv_i)}')
print(f'M: {M}')
#
x_vals = np.linspace(0.5, 3, 100)
slope = -w[0] / w[1]
intercept_margin_up = (1 - b) / w[1]
intercept_margin_down = (-1 - b) / w[1]
intercept = -b / w[1]
plt.plot(x_vals, slope * x_vals + intercept, 'k-')
plt.plot(x_vals, slope * x_vals + intercept_margin_up, 'k--')
plt.plot(x_vals, slope * x_vals + intercept_margin_down, 'k--')
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='b', label='class 1')
plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color='g', label='class -1')
plt.scatter(x_train[sv_i][:, 0], x_train[sv_i][:, 1], color='r', label='sv')
plt.scatter(x_train[bsv_i][:, 0], x_train[bsv_i][:, 1], color='pink', label='bsv')
plt.legend()
plt.savefig(f'x_train_g_{c}.png')
plt.close()


cm = confusion_matrix(y_train, pred)
ConfusionMatrixDisplay(cm, display_labels=['-1', '1']).plot()
plt.show()


x = generate_ring_data(n // 4, 3 * n // 4)
y = []
y.extend([-1] * (n // 4))
y.extend([1] * (3 * n // 4))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=52)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train[58] *= -1
y_train[1] *= -1
y_train[6] *= -1
#
n_train = len(y_train)
constraints = {
    'type': 'eq',
    'fun': lambda alpha: np.dot(alpha, y_train),
}
bounds = [(0, c) for _ in range(n_train)]
alpha0 = np.zeros(n_train)
result = minimize(fun=wolf_b, x0=alpha0, bounds=bounds, constraints=constraints, args=(x_train, y_train, sig,))
alpha = result.x
sv_i = (alpha > tol) * (alpha < c - tol)
bsv_i = (alpha >= c - tol) * (alpha <= c + tol)
b = np.mean(y_train[sv_i] - (alpha[sv_i] * y_train[sv_i]) @ k(sig, x_train[sv_i], x_train[sv_i]))
pred = predict_b(x_train, x_train[sv_i], y_train[sv_i], alpha[sv_i], b, sig)
M = compute_margin(alpha[sv_i], y_train[sv_i], x_train[sv_i], sig)
print(f'Сигма квадрат: {sig ** 2}')
print(f'Число опорных векторов: {sum(sv_i)}')
print(f'Число связанных опорных векторов: {sum(bsv_i)}')
print(f'M: {M}')
#
plot_svm_nonlinear_decision(x_train[sv_i], y_train[sv_i], alpha[sv_i], b, sig, bounds=[(-1, 1), (-1, 1)])
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='b', label='class 1')
plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color='g', label='class -1')
plt.scatter(x_train[sv_i][:, 0], x_train[sv_i][:, 1], color='r', label='sv')
plt.scatter(x_train[bsv_i][:, 0], x_train[bsv_i][:, 1], color='pink', label='bsv')
plt.legend()
plt.savefig(f'x_train_b_{c}.png')
plt.close()

cm = confusion_matrix(y_train, pred)
ConfusionMatrixDisplay(cm, display_labels=['-1', '1']).plot()
plt.show()



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
data = pd.read_csv(url, delim_whitespace=True, header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == 1, 1, -1)  # Преобразование меток: 1 -> +1, 2 -> -1
print(x)
X_extended = create_features(X)
x_train, x_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=47)
y_train = np.array(y_train)
y_test = np.array(y_test)
#
n_train = len(y_train)
constraints = {
    'type': 'eq',
    'fun': lambda alpha: np.dot(alpha, y_train),
}
bounds = [(0, c) for _ in range(n_train)]
alpha0 = np.zeros(n_train)
result = minimize(fun=wolf_b, x0=alpha0, bounds=bounds, constraints=constraints, args=(x_train, y_train, sig,))
alpha = result.x
sv_i = (alpha > tol) * (alpha < c - tol)
bsv_i = (alpha >= c - tol) * (alpha <= c + tol)
b = np.mean(y_train[sv_i] - (alpha[sv_i] * y_train[sv_i]) @ k(sig, x_train[sv_i], x_train[sv_i]))
pred = predict_b(x_test, x_train[sv_i], y_train[sv_i], alpha[sv_i], b, sig)
M = compute_margin(alpha[sv_i], y_train[sv_i], x_train[sv_i], sig)
print(f'Сигма квадрат: {sig ** 2}')
print(f'Число опорных векторов: {sum(sv_i)}')
print(f'Число связанных опорных векторов: {sum(bsv_i)}')
print(f'M: {M}')
#
cm = confusion_matrix(y_test,   pred)
ConfusionMatrixDisplay(cm, display_labels=['-1', '1']).plot()
plt.show()

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
data = pd.read_csv(url, delim_whitespace=True, header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == 1, 1, -1)  # Преобразование меток: 1 -> +1, 2 -> -1
X_extended = create_features(X)

x_train, x_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=47)
x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis=0)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

best_score = 0
best_params = None
#
for c_try in np.linspace(0.1, 10, 25):
    for sig_try in np.linspace(0.01, 10, 25):
        score = cross_val_score_svm(x_train, y_train, c_try, sig_try)
        print(f"c={c_try}, sig={sig_try} -> CV score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = (c_try, sig_try)
#
print(f"\nЛучшая пара: c={best_params[0]}, sig={best_params[1]} с точностью {best_score:.4f}")

pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)

explained = pca.explained_variance_ratio_
total_explained = np.sum(explained)

print(f"Доля объяснённой дисперсии: {total_explained:.4f} ({explained[0]:.4f} + {explained[1]:.4f})")


c = 2.1625
sig = 0.42625
n_train = len(y_train)
constraints = {
    'type': 'eq',
    'fun': lambda alpha: np.dot(alpha, y_train),
}
bounds = [(0, c) for _ in range(n_train)]
alpha0 = np.zeros(n_train)
result = minimize(fun=wolf_b, x0=alpha0, bounds=bounds, constraints=constraints, args=(x_train, y_train, sig,))
alpha = result.x
sv_i = (alpha > tol) * (alpha < c - tol)
bsv_i = (alpha >= c - tol) * (alpha <= c + tol)
b = np.mean(y_train[sv_i] - (alpha[sv_i] * y_train[sv_i]) @ k(sig, x_train[sv_i], x_train[sv_i]))
pred = predict_b(x_test, x_train[sv_i], y_train[sv_i], alpha[sv_i], b, sig)
M = compute_margin(alpha[sv_i], y_train[sv_i], x_train[sv_i], sig)
print(f'Сигма квадрат: {sig ** 2}')
print(f'Число опорных векторов: {sum(sv_i)}')
print(f'Число связанных опорных векторов: {sum(bsv_i)}')
print(f'M: {M}')
#
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=['-1', '1']).plot()
plt.show()



