import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    # TODO: Implement data collection logic
    data = pd.read_csv('data.csv')  # File dữ liệu
    
    # TODO: Implement data analysis logic
    print(data.describe())  # Hiển thị thống kê mô tả
    
    # TODO: Implement prediction logic
    X = data[['feature1', 'feature2']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # TODO: Implement visualization logic
    plt.scatter(X_test['feature1'], y_test, color='blue', label='Thực tế')
    plt.scatter(X_test['feature1'], predictions, color='red', label='Dự đoán')
    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
