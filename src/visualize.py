import matplotlib.pyplot as plt  
import matplotlib.pyplot as plt
import seaborn as sns


def show_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot(kind='bar')
    plt.show()

def show_distrebution_of_categatial_column_valuse(df, column_name):
    graph = sns.countplot(x=column_name, data=df)
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
    plt.show()



    # visualize = Visualize()
    # visualize.show_distrebution_of_categatial_column_valuse(train_data, 'color')
    # visualize.show_distrebution_of_categatial_column_valuse(test_data, 'color')
