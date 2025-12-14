history = np.array(problem.history)
generation = np.arange(1, len(history) + 1)

history = -history

plt.figure(figsize=(10, 6))
plt.plot(generation, history[:, 0], label='Polyphenol Target Value', color='blue')
plt.plot(generation, history[:, 1], label='Chlorogenic Acid Target Value', color='red')
plt.xlabel('Generation')
plt.ylabel('Objective Function Value')
plt.title('Genetic Algorithm Optimization Process: Objective Function Value vs Generations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
