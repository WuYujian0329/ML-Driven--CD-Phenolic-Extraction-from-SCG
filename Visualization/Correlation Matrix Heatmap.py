import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(file_path)
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.tight_layout()
plt.show()
