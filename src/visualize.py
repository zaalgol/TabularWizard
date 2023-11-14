import matplotlib.pyplot as plt  
import matplotlib.pyplot as plt
import seaborn as sns


def show_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        print("No missing values found!")
        return

    missing.sort_values(inplace=True)
    missing.plot(kind='bar')
    plt.show()


# the clolumn has a few possible values
def show_distrebution_of_categatial_column_valuse(df, column_name):
    graph = sns.countplot(x=column_name, data=df)
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
    plt.show()


# the clolumn has a a lot of possible values
def show_distribution_of_numeric_column_values(df, column_name):
    """
    Display the distribution of a numeric column using a histogram and KDE.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: The column to visualize

    Returns: None
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column_name], kde=True)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def plot_all_correlation(df):
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.show()

def plot_correlation_two_columns(df, col1, col2):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=col1, y=col2, data=df)
    plt.title(f'Correlation between {col1} and {col2}')
    plt.show()

def plot_correlation_one_vs_others(df, column_name):
    correlations = df.corr()[column_name].drop(column_name)
    correlations.sort_values(inplace=True)
    correlations.plot(kind='bar', figsize=(8, 6))
    plt.title(f'Correlation of {column_name} with other columns')
    plt.ylabel('Correlation Coefficient')
    plt.show()
