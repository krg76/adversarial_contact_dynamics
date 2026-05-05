import pandas as pd
import matplotlib.pyplot as plt

def create_training_plots(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # --- CONFIGURATION SECTION ---
    # Edit these values to change the labels, titles, and axes names
    
    # Configuration for Plot 1: Losses
    config1 = {
        'x': 'iteration',
        'y1': 'd_loss',
        'y2': 'g_loss',
        'label1': 'Discriminator Loss',
        'label2': 'Generator Loss',
        'title': 'Training Losses',
        'xlabel': 'Epoch',
        'ylabel': 'Loss Value',
        'filename': 'loss_plot.png'
    }

    # Configuration for Plot 2: Metrics in Log Space
    config2 = {
        'x': 'iteration',
        'y1': 'k1',
        'y2': 'd1',
        'label1': 'Stiffness Parameter',
        'label2': 'Damping Parameter',
        'title': 'Contact Parameter Evolution (Log Scale)',
        'xlabel': 'Epochs',
        'ylabel': 'Value (Log)',
        'filename': 'metrics_log_plot.png'
    }

    # --- PLOTTING SECTION ---

    # Plot 1: d_loss and g_loss
    plt.figure(figsize=(10, 5))
    plt.plot(df[config1['x']], df[config1['y1']], label=config1['label1'], color='#0055d4', marker='o')
    plt.plot(df[config1['x']], df[config1['y2']], label=config1['label2'], color='#d48800', marker='x')
    plt.title(config1['title'])
    plt.xlabel(config1['xlabel'])
    plt.ylabel(config1['ylabel'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(config1['filename'])
    plt.show()

    # Plot 2: k1 and d1 in log space
    plt.figure(figsize=(10, 5))
    plt.plot(df[config2['x']], df[config2['y1']], label=config2['label1'], color='#0055d4', marker='o')
    plt.plot(df[config2['x']], df[config2['y2']], label=config2['label2'], color='#d48800', marker='x')
    plt.yscale('log') # Sets the y-axis to log space
    plt.title(config2['title'])
    plt.xlabel(config2['xlabel'])
    plt.ylabel(config2['ylabel'])
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config2['filename'])
    plt.show()

# Run the script
create_training_plots('gan_results/training_history.csv')