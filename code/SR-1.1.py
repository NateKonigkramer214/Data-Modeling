#By Nathan Konigkramer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#!TO LOAD DATA
def load_data(file_name):
    return pd.read_excel(file_name)

#!TO VISUALISE DATA
def Visualise_Data(data):
    return sns.pairplot(data)

if __name__ == "__main__":
    data = load_data('Net_Worth_Data.xlsx')
    Visualise_Data(data)
    plt.show()
    

    
    