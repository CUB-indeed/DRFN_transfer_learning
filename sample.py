import numpy as np
import matplotlib.pyplot as plt

def generate_random_values(mean, std, n=10):
    """
    Generate n random values centered around a given mean with a specified standard deviation.
    
    Parameters:
        mean (float): The mean value.
        std (float): The standard deviation.
        n (int): Number of random values to generate. Default is 10.
    
    Returns:
        list: A list of n random values.
    """
    return list(np.random.normal(loc=mean, scale=std, size=n))

def plot_two_boxplots(random_values1, random_values2, title1="Box Plot 1", title2="Box Plot 2"):
    """
    Plot two box plots in subplots with inverted axes (y-axis as values, x-axis labeled as RMSE).
    
    Parameters:
        random_values1 (list): The first set of random values.
        random_values2 (list): The second set of random values.
        title1 (str): Title for the first subplot. Default is "Box Plot 1".
        title2 (str): Title for the second subplot. Default is "Box Plot 2".
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    # Plot first boxplot
    axes[0].boxplot(random_values1, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red"))
    axes[0].set_title(title1)
    axes[0].set_ylabel("Values")
    axes[0].set_xlabel("RMSE")
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot second boxplot
    axes[1].boxplot(random_values2, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor="lightgreen", color="green"),
                    medianprops=dict(color="red"))
    axes[1].set_title(title2)
    axes[1].set_ylabel("Values")
    axes[1].set_xlabel("RMSE")
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plots
    plt.show()

# Example usage
mean1, std1 = 718.80, 2.8
mean2, std2 = 146.78, 1.3

random_values1 = generate_random_values(mean1, std1, n=10)
random_values2 = generate_random_values(mean2, std2, n=10)

plot_two_boxplots(random_values1, random_values2, 
                  title1="RMSE Box Plot for target-1", 
                  title2="RMSE Box Plot for target-2")