from pathlib import Path

import matplotlib.pyplot as plt


output = Path(__file__).resolve().parent.parent / "Output"
output.mkdir(exist_ok=True)
output_path = output


def plot_config():
    """Apply shared Matplotlib defaults for the project."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (7, 4)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 11
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    #locale.setlocale(locale.LC_NUMERIC, 'sv_SE.UTF-8')  
    plt.rcParams['axes.formatter.use_locale'] = True
    return output
    
