from sklearn.ensemble import GradientBoostingRegressor
import joblib
import csv
import numpy as np

train_num = 70000
feature = []
with open('sampled_data_processed.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    headline = next(reader)
    for line in reader:
#         latency.append(float(line[0]))
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
    if x <= 10:
        error_distribution[0] += 1
    elif 10 < x <= 20:
        error_distribution[1] += 1
    elif 20 < x <= 30:
        error_distribution[2] += 1
    elif 30 < x <= 40:
        error_distribution[3] += 1
    elif 40 < x <= 50:
        error_distribution[4] += 1
    elif 50 < x <= 60:
        error_distribution[5] += 1
    elif 60 < x <= 70:
        error_distribution[6] += 1
    elif 70 < x <= 80:
        error_distribution[7] += 1
    elif 80 < x <= 90:
        error_distribution[8] += 1
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