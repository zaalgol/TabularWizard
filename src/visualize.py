import matplotlib.pyplot as plt    

class Visualize:

    def show_missing(self, df):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot(kind='bar')
        plt.show()
