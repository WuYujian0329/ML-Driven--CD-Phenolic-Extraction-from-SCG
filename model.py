import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

file_path = r"augmented_data_with_original_and_noise.xlsx"
df = pd.read_excel(file_path)

X_raw = df.iloc[:, 0:4].values
y_raw = df.iloc[:, -2:].values

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

def purelin(x):
    return x

def forward_pass(X, weights):
    W1, b1, W2, b2 = weights
    Z1 = X @ W1 + b1
    A1 = tansig(Z1)
    Z2 = A1 @ W2 + b2
    A2 = purelin(Z2)
    return A1, A2

input_dim = 4
hidden_dim = 15
output_dim = 2

def residuals(flat_weights, X, y, input_dim, hidden_dim, output_dim):
    W1 = flat_weights[:input_dim * hidden_dim].reshape(input_dim, hidden_dim)
    b1 = flat_weights[input_dim * hidden_dim : input_dim * hidden_dim + hidden_dim]
    start = input_dim * hidden_dim + hidden_dim
    W2 = flat_weights[start : start + hidden_dim * output_dim].reshape(hidden_dim, output_dim)
    b2 = flat_weights[start + hidden_dim * output_dim:]
    weights = (W1, b1, W2, b2)
    _, y_pred = forward_pass(X, weights)
    return (y_pred - y).ravel()

np.random.seed(42)
W1_init = np.random.randn(input_dim, hidden_dim) * 0.1
b1_init = np.zeros(hidden_dim)
W2_init = np.random.randn(hidden_dim, output_dim) * 0.1
b2_init = np.zeros(output_dim)
flat_init = np.concatenate([W1_init.ravel(), b1_init, W2_init.ravel(), b2_init])

result = least_squares(
    residuals,
    flat_init,
    args=(X_train, y_train, input_dim, hidden_dim, output_dim),
    method='trf',
    max_nfev=200
)

flat_opt = result.x
W1 = flat_opt[:input_dim * hidden_dim].reshape(input_dim, hidden_dim)
b1 = flat_opt[input_dim * hidden_dim : input_dim * hidden_dim + hidden_dim]
start = input_dim * hidden_dim + hidden_dim
W2 = flat_opt[start : start + hidden_dim * output_dim].reshape(hidden_dim, output_dim)
b2 = flat_opt[start + hidden_dim * output_dim:]
trained_weights = (W1, b1, W2, b2)

_, y_pred_scaled = forward_pass(X_test, trained_weights)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

residuals_eval = y_true - y_pred
RSS = np.sum(residuals_eval ** 2, axis=0)
TSS = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
R2 = 1 - RSS / TSS

RSE = np.sqrt(RSS / (len(y_true) - 2))

print("\n=== Model Evaluation ===")
print(f"R2 for Polyphenols: {R2[0]:.4f}, RSE: {RSE[0]:.4f}")
print(f"R2 for Chlorogenic Acid: {R2[1]:.4f}, RSE: {RSE[1]:.4f}")

class DualTargetOptimization(Problem):
    def __init__(self, scaler, weights):
        super().__init__(n_var=4, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        self.scaler = scaler
        self.weights = weights

    def _evaluate(self, X, out, *args, **kwargs):
        _, pred_scaled = forward_pass(X, self.weights)
        pred = y_scaler.inverse_transform(pred_scaled)
        polyphenols = pred[:, 0]
        chlorogenic = pred[:, 1]
        out["F"] = -np.column_stack([polyphenols, chlorogenic])

problem = DualTargetOptimization(x_scaler, trained_weights)

algorithm = NSGA2(
    pop_size=100,
    sampling=LHS(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 200)

res = minimize(problem,
               algorithm,
               termination,
               seed=2025,
               save_history=True,
               verbose=True)

X_opt = x_scaler.inverse_transform(res.X)
Y_opt = -res.F

print("\n=== NSGA-II Optimal Solution Set (Top 10) ===")
for i in range(min(10, len(X_opt))):
    print(f"Combination {i+1}:")
    print(f"  Input: {X_opt[i]}")
    print(f"  Polyphenol content: {Y_opt[i, 0]:.4f}, Chlorogenic Acid: {Y_opt[i, 1]:.4f}")


