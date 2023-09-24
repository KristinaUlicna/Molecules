import numpy as np
import matplotlib.pyplot as plt


def prepare_cdf(
    values: list[float], 
    bins: int = 50, 
) -> tuple[list[float], list[float]]:
    """Helper function to plot the CDF plots."""
    count, bins_count = np.histogram(values, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf
    

def plot_distributions(
    pIC50_values: list[list[float]], 
    **kwargs
) -> None:
    """Plot the histogram & CDF of train, (valid) and test set logD values."""
    fig, axes = plt.subplots(1, 2, **kwargs)
    names = ["Train", "Valid", "Test"]

    # Plot the histogram
    for i in range(len(pIC50_values)):
        axes[0].hist(pIC50_values[i], label=f"{names[i]} Data")
    
    axes[0].set_title("Dataset pIC50 value distribution")
    axes[0].set_xlabel("pIC50 Value")
    axes[0].set_ylabel("Compounds")
    axes[0].legend()

    # Plot the CDF
    for i in range(len(pIC50_values)):
        bc, cdf = prepare_cdf(values=pIC50_values[i])
        axes[1].plot(bc, cdf, label=f"{names[i]} Data")

    axes[1].set_title("Dataset pIC50 value CDF curve")
    axes[1].set_ylabel("Cumulative Density")
    axes[1].set_xlabel("pIC50 Value")
    axes[1].legend()
    
    return fig


def plot_training_curves(
    train_losses: list[float], 
    valid_losses: list[float], 
    infer_loss: float, 
    **kwargs,
) -> None:
    fig = plt.figure(**kwargs)
    num_epochs = len(train_losses)
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.plot([0, num_epochs], [infer_loss, infer_loss], label="Infer")
    plt.title(f"Training for {num_epochs}-epoch GAT model")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    return fig
    