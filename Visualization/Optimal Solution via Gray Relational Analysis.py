import numpy as np

ideal_solution = np.array([max(Y_opt[:, 0]), max(Y_opt[:, 1])])

def gray_relational_analysis(X, ideal_solution, rho=0.5):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    delta_ideal = np.abs(X - ideal_solution)
    delta_0 = np.abs(ideal_solution - ideal_solution)
    
    denominator = np.sum(delta_0 + rho * delta_ideal, axis=1)
    
    return 1 / denominator

gray_association = gray_relational_analysis(Y_opt, ideal_solution)

best_index = np.argmax(gray_association)
best_solution = X_opt[best_index]
best_Y = Y_opt[best_index]

print("\n=== Optimal Solution (Gray Relational Analysis) ===")
print(f"Input: {best_solution}")
print(f"Polyphenol Content: {best_Y[0]:.4f}, Chlorogenic Acid: {best_Y[1]:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(Y_opt[:, 0], Y_opt[:, 1], c='blue', edgecolors='black', s=40, label="Optimized Solutions")
plt.scatter(best_Y[0], best_Y[1], c='red', s=100, marker='*', label="Optimal Solution")
plt.xlabel("Polyphenol Content")
plt.ylabel("Chlorogenic Acid Content")
plt.title("Optimal Solution and Pareto Front")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
