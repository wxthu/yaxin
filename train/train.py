from sklearn.ensemble import GradientBoostingRegressor
import joblib
import csv
import numpy as np

train_num = 104000
feature = []
with open('sampled_data_processed.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    headline = next(reader)
    for line in reader:
        feature.append([int(line[3]), int(line[6]), float(line[7]), float(line[8]), float(line[0])])

feature = np.array(feature)
np.random.shuffle(feature)
latency = feature[:,4:]
feature = feature[:,:4]

x_train = feature[:train_num, :]
y_train = latency[:train_num, :]
x_test = feature[train_num:, :]
y_test = latency[train_num:, :]
y_test = y_test.squeeze()

reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

error = abs(y_pred - y_test) / y_test * 100

error_distribution = [0 for _ in range(10)]
for x in error:
    index= int(x/10)
    if index <= 9:
        error_distribution[int(x/10)] += 1
    else: 
        error_distribution[9] += 1
        
s = sum(error_distribution)
percentage = []
for x in error_distribution:
    percentage.append(x/s * 100)
print('Test error distribution as follows: ')
print(error_distribution)
print('Converting to percentage: ')
print(percentage)

# save trained regressor
joblib.dump(reg, './lat_pred.pkl')