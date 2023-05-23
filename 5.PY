import numpy as np
import pandas as pd

data  = pd.read_csv('data.csv')

data

data.describe()

data.info()

data["age"].mean()

data["age"].count()

data["age"].median()

data["age"].max()

data["age"].min()

data["age"].std()

data["income"].nunique()

data.groupby(["income","age"]).count()

data.groupby(["income","age"]).mean()

data.groupby(["income","age"]).max()

data.groupby(["income","age"]).median()

data.groupby(["income","age"]).std()

#2

import pandas as pd
import numpy as np
import seaborn as sns

data = sns.load_dataset('iris')
print("Original dataset")
data

data.head()

data.describe()

np.percentile(data["sepal_length"],25)

data["sepal_length"].std()

data["sepal_length"].mean()

data["sepal_length"].median()

group = data.groupby("species")

iris_versicolor = group.get_group('versicolor')

iris_versicolor

iris_versicolor.describe()

iris_versicolor=group.get_group('setosa')

iris_versicolor

iris_versicolor.describe()
