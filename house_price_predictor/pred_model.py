import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
house=pd.read_csv('homeprices_multi.csv')
x=house['area(Sq.feet)']
y=house.price
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
reg=LinearRegression()
reg.fit(pd.DataFrame(x_train),pd.DataFrame(y_train))
with open('house_data.pkl','wb') as f:
    pickle.dump(reg, f)
print("Model trained and saved as 'house_data.pkl'")