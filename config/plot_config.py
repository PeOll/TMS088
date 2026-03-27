import matplotlib.pyplot as plt


def plot_config():
    """Apply shared Matplotlib defaults for the project."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 11
    
