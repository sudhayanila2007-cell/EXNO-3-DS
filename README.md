## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="611" height="365" alt="Screenshot 2026-08-18 153929" src="https://github.com/user-attachments/assets/fde95501-0bf9-4999-bce2-0345245c5e31" />

```
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="638" height="246" alt="Screenshot 2026-08-18 153947" src="https://github.com/user-attachments/assets/0e7b2880-2a96-4d11-928f-8d8e87d86e58" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="673" height="377" alt="Screenshot 2026-08-18 153955" src="https://github.com/user-attachments/assets/a4e8ebd3-c853-40f3-b45c-a0bc82d6a700" />

```
# Label Encoder ( orders in alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="659" height="367" alt="Screenshot 2026-08-18 154002" src="https://github.com/user-attachments/assets/c46f5c8e-50c8-4365-a7fe-4f499d14de7e" />

```
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="718" height="365" alt="Screenshot 2026-08-18 154009" src="https://github.com/user-attachments/assets/1ffae598-9000-4553-872c-c10bbb48b864" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="966" height="376" alt="Screenshot 2026-08-18 154021" src="https://github.com/user-attachments/assets/518f36a1-2454-4c96-8545-cd851840693f" />

```
pip install --upgrade category_encoders
```
<img width="1253" height="505" alt="Screenshot 2026-08-18 154055" src="https://github.com/user-attachments/assets/c1a1f5ef-52bf-4be9-b137-6c4610d075e8" />

```
# BINARY ENCODER
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="733" height="357" alt="Screenshot 2026-08-18 154103" src="https://github.com/user-attachments/assets/f2bd4c41-2c80-4ffc-b986-ab1513544c3d" />
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="983" height="365" alt="Screenshot 2026-08-18 154111" src="https://github.com/user-attachments/assets/216755f7-5d72-4674-a3c3-9b11e82aa233" />

```
# MEAN ENCODING
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="839" height="360" alt="Screenshot 2026-08-18 154118" src="https://github.com/user-attachments/assets/885c66c1-f509-4490-8e18-c190bd3a6b03" />

```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="1066" height="458" alt="Screenshot 2026-08-18 154127" src="https://github.com/user-attachments/assets/f989a7df-0e13-41de-863d-54634f223d74" />

```
df.skew()
```
<img width="941" height="136" alt="Screenshot 2026-08-18 154135" src="https://github.com/user-attachments/assets/e00a0e9e-7953-4d83-bd6f-6f0989b5d98b" />

```
# 1. LOG TRANSFORMATION 
np.log(df["Highly Positive Skew"])
```
<img width="780" height="268" alt="Screenshot 2026-08-18 154143" src="https://github.com/user-attachments/assets/4ad0f46e-209d-4d5e-8256-d2d2e3746cf1" />

```
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="864" height="282" alt="Screenshot 2026-08-18 154150" src="https://github.com/user-attachments/assets/890b9596-5dfa-4854-a9cb-d15c54e1672e" />

```
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
```
<img width="994" height="279" alt="Screenshot 2026-08-18 154157" src="https://github.com/user-attachments/assets/61327e7e-7fcd-4bcd-9502-2829b0c40ae8" />

```
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
```
<img width="927" height="286" alt="Screenshot 2026-08-18 154204" src="https://github.com/user-attachments/assets/a81a0cf7-de12-4670-b4d8-54592f192762" />

```
# POWER TRANSFORMATIONS 
# BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="1125" height="448" alt="Screenshot 2026-08-18 154214" src="https://github.com/user-attachments/assets/733b1e6b-b521-405f-8622-133b4b0c8bef" />

```
df.skew()
```
<img width="808" height="152" alt="Screenshot 2026-08-18 154415" src="https://github.com/user-attachments/assets/806cd910-32d1-4bb4-95da-605ca5ca5fd9" />

```
# YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="914" height="183" alt="Screenshot 2026-08-18 154421" src="https://github.com/user-attachments/assets/bc5d1370-7c87-4369-9292-da3947232fbc" />

```
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="1337" height="459" alt="Screenshot 2026-08-18 154436" src="https://github.com/user-attachments/assets/683d5673-9ab6-4e3f-948d-0a21d0715b26" />

```
import seaborn as sns 
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT 
plt.show()
```
<img width="1307" height="547" alt="Screenshot 2026-08-18 154453" src="https://github.com/user-attachments/assets/a38b06cb-48f2-45c3-83a4-8f0188fb8353" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```
<img width="1155" height="554" alt="Screenshot 2026-08-18 154506" src="https://github.com/user-attachments/assets/8dae7603-da0c-4274-90fe-3d62d26b6770" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="1133" height="557" alt="Screenshot 2026-08-18 154519" src="https://github.com/user-attachments/assets/1bea944b-dc84-4d6e-928a-0ccf64038dc2" />



# RESULT:
      Thus ,the given data Feature Encoding and Transformation process executed successfully.

       
