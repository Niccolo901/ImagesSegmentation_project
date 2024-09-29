import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class FeaturePlotter:
    def __init__(self, csv_path):
        """
        Initializes the FeaturePlotter class.
        
        Parameters:
            csv_path (str): Path to the features_output.csv file.
        """
        self.csv_path = csv_path
        self.df = None
    
    def load_data(self):
        """
        Loads the CSV file into a DataFrame.
        """
        self.df = pd.read_csv(self.csv_path)
    
    def list_available_features(self):
        """
        Lists all available features (columns) in the dataset.
        
        Returns:
            list: List of feature names available in the dataset.
        """
        if self.df is not None:
            return self.df.columns.tolist()
        else:
            print("Data not loaded. Please call load_data() first.")
            return []
    
    def plot_features_distribution_with_intersection(self, feature_names, label_column='Label', bins=30):
        """
        Plots the distribution of the selected features and finds the intersection point(s).
        
        Parameters:
            feature_names (list): List of feature names to plot.
            label_column (str): Name of the column that contains labels for different classes (default is 'Label').
            bins (int): Number of bins for the histogram (default is 30).
        """
        if self.df is None:
            print("Data not loaded. Please call load_data() first.")
            return
        
        for feature_name in feature_names:
            if feature_name not in self.df.columns:
                print(f"Feature '{feature_name}' not found in the dataset.")
                continue

            plt.figure(figsize=(12, 6))
            sns.kdeplot(data=self.df, x=feature_name, hue=label_column, fill=True, common_norm=False, palette='viridis')

            # Calculate KDEs for each label class
            kde_0 = stats.gaussian_kde(self.df[self.df[label_column] == 0][feature_name])
            kde_1 = stats.gaussian_kde(self.df[self.df[label_column] == 1][feature_name])

            # Generate values for the x-axis
            x = np.linspace(self.df[feature_name].min(), self.df[feature_name].max(), 1000)
            kde_0_values = kde_0(x)
            kde_1_values = kde_1(x)

            # Find the intersection point(s)
            intersections = np.where(np.diff(np.sign(kde_0_values - kde_1_values)))[0]

            if len(intersections) > 0:
                intersection_point = x[intersections[-1]]  # Take the last (rightmost) intersection point
                plt.axvline(intersection_point, color='red', linestyle='--')
                plt.scatter([intersection_point], [kde_0(intersection_point)], color='red')  # Mark intersection
                plt.text(intersection_point, kde_0(intersection_point), f'  {intersection_point:.2f}', color='red')
                print(f"Intersection point found for {feature_name}: {intersection_point}")
            else:
                print(f"No intersection point found for {feature_name}.")
            
            plt.title(f'Distribution of {feature_name}')
            plt.xlabel(feature_name)
            plt.ylabel('Density')
            plt.grid(True)
            plt.show()

    def plot_tsne(self, feature_names, label_column='Label'):
        """
        Plots a t-SNE visualization of the selected features.
        
        Parameters:
            feature_names (list): List of feature names to include in the t-SNE plot.
            label_column (str): Name of the column that contains labels for different classes (default is 'Label').
        """
        if self.df is None:
            print("Data not loaded. Please call load_data() first.")
            return

        # Check if all feature names are in the DataFrame
        for feature_name in feature_names:
            if feature_name not in self.df.columns:
                print(f"Feature '{feature_name}' not found in the dataset.")
                return

        # Extract the features and labels
        X = self.df[feature_names].values
        y = self.df[label_column].values

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
        X_tsne = tsne.fit_transform(X_scaled)

        # Plot the t-SNE results
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', legend='full')
        plt.title('t-SNE Visualization of Selected Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.show()
