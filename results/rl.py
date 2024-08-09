# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    return pd.read_csv(file_path)

def create_reward_plot(base_path_high, base_path_low):
    df1 = read_csv(base_path_high + 'reward.csv')
    df2 = read_csv(base_path_low + 'reward.csv')

    plt.figure(figsize=(12, 6))
    plt.plot(df1['Step'], df1['Value'], label='High Entropy Run')
    plt.plot(df2['Step'], df2['Value'], label='Low Entropy Run')
    plt.xlabel('Mutations')
    plt.ylabel('Reward Value')
    plt.title('Reward Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Add vertical lines at powers of 10
    for j in range(int(np.log10(df1['Step'].min())), int(np.log10(df1['Step'].max()))+1):
        plt.axvline(10**j, color='gray', linestyle='--', alpha=0.5)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("reward_comparison.png", dpi=300)
    plt.close()

def create_other_plots(metrics, base_path_high, base_path_low):
    n_metrics = len(metrics)
    fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics), sharex=True)
    fig.suptitle('Comparison of Training Runs', fontsize=16)

    for i, (metric, ylabel, filename) in enumerate(metrics):
        try:
            df1 = read_csv(base_path_high + filename)
            df2 = read_csv(base_path_low + filename)

            axs[i].plot(df1['Step'], df1['Value'], label='High Entropy Run')
            axs[i].plot(df2['Step'], df2['Value'], label='Low Entropy Run')
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f"{metric.replace('_', ' ').title()}")
            axs[i].legend()
            axs[i].grid(True)
            axs[i].set_xscale('log')
            
            # Add vertical lines at powers of 10
            for j in range(int(np.log10(df1['Step'].min())), int(np.log10(df1['Step'].max()))+1):
                axs[i].axvline(10**j, color='gray', linestyle='--', alpha=0.5)
            
            axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            print(f"Added comparison plot for {metric}")
        except Exception as e:
            print(f"Error creating plot for {metric}: {str(e)}")
            axs[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')

    axs[-1].set_xlabel('Gradient Steps')
    plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("other_metrics_comparison.png", dpi=300)
    plt.close()

base_path_high = '/root/projects/rl-enzyme-engineering/results/data/sequence_rl_high_ent/training/'
base_path_low = '/root/projects/rl-enzyme-engineering/results/data/sequence_rl_low_ent/training/'

# Create reward plot
create_reward_plot(base_path_high, base_path_low)
print("Reward comparison plot has been created and saved as 'reward_comparison.png'.")

# Create other metrics plots
other_metrics = [
    ('entropy', 'Entropy Value', 'site_picker_ent.csv'),
    ('site_picker_loss', 'Loss Value', 'site_picker_loss.csv'),
    ('value_loss', 'Loss Value', 'value_function.csv')
]

create_other_plots(other_metrics, base_path_high, base_path_low)
print("Other metrics comparison plot has been created and saved as 'other_metrics_comparison.png'.")