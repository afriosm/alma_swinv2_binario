import matplotlib.pyplot as plt

def plot_learning_curves(history, title="Learning Curves"):
    epochs = history["epoch"]

    # Loss (train_loss vs val_log_loss)
    plt.figure(figsize=(7,4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_log_loss"], label="val_log_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1 macro
    plt.figure(figsize=(7,4))
    plt.plot(epochs, history["train_f1_macro"], label="train_f1_macro")
    plt.plot(epochs, history["val_f1_macro"], label="val_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 macro")
    plt.title(f"{title} - F1 Macro")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Balanced accuracy
    plt.figure(figsize=(7,4))
    plt.plot(epochs, history["train_balanced_acc"], label="train_balanced_acc")
    plt.plot(epochs, history["val_balanced_acc"], label="val_balanced_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title(f"{title} - Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()