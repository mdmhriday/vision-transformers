import os
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, save_dirs="./logs"):
        self.save_dirs = save_dirs
        os.makedirs(self.save_dirs, exist_ok=True)

    def plot(self, name, training_values, val_values=None):
        epochs = range(1, len(train_losses)+1)
        plt.Figure(figsize=(8, 5))

        plt.plot(epochs, train_values, label=f"Train {name}", marker="o", linestyle="--", color="blue")
        if val_values:
            plt.plot(epochs, val_values, label=f"Val {name}", marker="o", linestyle="--", color="red")
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.title(f"{name} Over Time")
        plt.Legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, f"{name.lower()}_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved: {save_path}")