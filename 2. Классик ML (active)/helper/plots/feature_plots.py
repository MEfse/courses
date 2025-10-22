import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class BuildHist:
    def __init__(self, series_dict):
        self.data = series_dict
        self.linewidth = 0.2
        self.edgecolor = 'red'

    def get_name(self):
        if type(self.data) == dict:
            xlabel = self.data[list(self.data.keys())[0]].name
        else:
            xlabel = self.data.name
        return xlabel

    def categorical_hist(self):
        xlabel = self.get_name()

        rows = []

        for name, series in self.data.items():
            values = series.value_counts()
            for category, count in values.items():
                rows.append((category, name, count))

        df = pd.DataFrame(rows, columns=['category', 'name', 'count'])

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, 
                    x='category',
                    y='count',
                    hue='name', 
                    linewidth=self.linewidth, 
                    edgecolor=self.edgecolor);

        first_key = list(self.data.keys())[0]

        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title("Categorical comparison")
        #plt.legend()
        plt.grid(True)

        plt.show()

    def numeric_plot(self):
        xlabel = self.get_name()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data,
                    bins='auto', 
                    linewidth=self.linewidth, 
                    edgecolor=self.edgecolor);

        first_key = list(self.data.keys())[0]

        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title("Numeric comparison")
        #plt.legend()
        plt.grid(True)

        plt.show()

    def numeric_boxplot(self):
        xlabel = self.get_name()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(self.data, orient="y");

        first_key = list(self.data.keys())[0]

        plt.xlabel(xlabel)
        #plt.ylabel('Count')
        plt.title("Numeric Boxplot")
        #plt.legend()
        plt.grid(True)

        plt.show()

    def matrix_multicollinearity(self, matrix, mask):
        plt.figure(figsize=(18, 9))
        ax = sns.heatmap(matrix[mask], annot=True, linewidths=0.01, cmap='crest');
        plt.grid(True)

        return ax