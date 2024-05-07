# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
from scipy import stats
import pandas as pd
df=pd.read_csv('/content/bmi.csv')
df.head()
```
![Screenshot 2024-05-04 085839](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/f34f06ab-90d9-4158-a4de-85da3dd7945c)

```
df.dropna()
```
![Screenshot 2024-05-04 085848](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/e1654674-e970-4abd-a11f-c1d21eb6fc42)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
```
![Screenshot 2024-05-04 085916](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/6c5aced7-436c-4efb-807f-3642784052a6)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Head','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-05-04 085928](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/22b2d3af-9acd-4247-a5c6-62ffb7497b9e)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-05-04 085937](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/9bc2565d-9e82-4947-902d-7e4075538fae)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-05-04 085946](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/6032f3be-17bc-416d-b50c-2cd2e8314582)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-05-04 085954](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/6b1d84b8-3221-4eff-bc76-d338902d8f35)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-05-04 090002](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/1824a244-a929-4c9d-94c4-3ec996c6afee)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![Screenshot 2024-05-04 090008](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/f8f41199-faec-4092-8f94-4ec0742243de)

```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-05-04 090018](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/c3df7338-903e-4ac6-800b-3597b68d7aa0)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head
```
![Screenshot 2024-05-04 090026](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/74eae356-3194-459e-9f55-7dde4b1acb51)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-05-04 090035](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/4d76e401-f386-42e1-ac0e-9d18733f87a3)

```
chi2, p, _, _=chi2_contingency(contingency_table)
print("Chi-Square Statistic: {chi2}")
print(f"P-value:{p}")
```
![Screenshot 2024-05-04 090043](https://github.com/Harsayazheni/Expt04-Introduction-to-Data-Science/assets/118708467/bbffe9d5-57ff-404e-a8bd-848e9528feba)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
