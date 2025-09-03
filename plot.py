import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('loss_entropy.csv')

# Define custom colors
colors = {
    "loss": ["blue", "Value"],
    "policy_loss": ["orange", "Cross Entropy"],
    "value_loss": ["green", "MSE Loss"],
    "entropy": ["red", "Entropy"]
}

# Loop through each metric
for col, properties in colors.items():
    plt.figure(figsize=(8, 6))
    plt.plot(df[col], label=col.replace("_", " ").title(), color=properties[0])
    plt.xlabel("Step")
    plt.ylabel(properties[1])
    plt.title(col.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 720)
    plt.ylim(df[col].min(), df[col].max())
    plt.show()
