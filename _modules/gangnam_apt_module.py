import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import os

def preprocessing():
    # 행정 구역을 integer로 변환
    # area = {'개포동': 0, '논현동': 1, '대치동': 2, '도곡동': 3, '삼성동': 4, '세곡동': 5,
    #         '수서동': 6, '신사동': 7, '압구정동': 8, '역삼동': 9, '일원동': 10, '청담동': 11}
    # 데이터 전처리
    from sklearn.preprocessing import PolynomialFeatures        #특성을 만들거나 전처리를 위해 제공되는 클래스

    data = pd.read_csv(f'{os.getcwd()}/_modules/apt.csv',
                            encoding='cp949', engine='python', index_col=False)
    #print(data)

    data = data.rename(columns={'연도': 'year',
                         '행정구역(동)': 'area',
                         '1㎡당 가격':'price'},
                    inplace=False) #inplace :True원본데이터 수정
    #print(data)

    data['price'] = data['price'].map(lambda x:str(x).replace(',', ''), na_action=None).astype('float64').apply(lambda x:x*3.3)

    return data



def apt_predict(predict_year, predict_dong):
    data = preprocessing()
    area = {'0': '개포동', '1': '논현동', '2': '대치동', '3': '도곡동', '4': '삼성동', '5': '세곡동',
     '6': '수서동', '7': '신사동', '8': '압구정동', '9': '역삼동', '10': '일원동', '11': '청담동'}

    condition = (data.area == area[predict_dong])
    year = data['year'].unique()
    price = data['price'][condition].to_numpy()

    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(year, price, random_state=40)
    train_input = train_input.reshape(-1,1)
    test_input = test_input.reshape(-1,1)

    train_poly = np.column_stack((train_input**2,train_input))
    test_poly = np.column_stack((test_input**2,test_input))

    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    score = lr.score(test_poly, test_target)
    years = range(predict_year-4, predict_year+1)
    #print(score)
    point = np.arange(years[0],years[-1]+1)
    #plt.plot(point, lr.coef_[0] * point ** 2 + lr.coef_[1] * point + lr.intercept_, c='skyblue')

    predict_prices = []
    for year in years:
        predict = lr.predict([[year**2,year]])
        predict_prices.append(int(predict))
        # if year!=predict_year:
        #     plt.scatter(year, predict, c='skyblue')
        # else:
        #     plt.scatter(year, predict, c='red', marker='^')
    predict_dong = area[predict_dong]
    print(f"{predict_year}년 {predict_dong}의 1 평당 가격은 {predict_prices[0]:.2f} 만원으로 예상됩니다. 정확도: {score*100:.2f}")
    #plt.show()

    return predict_prices, predict_dong, int(score*100)

def save_fig(predict_year, predict_prices):
    img = BytesIO()
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['font.size'] = 10

    years = range(predict_year-5, predict_year+1)
    for year, price in zip(years, predict_prices):
        if year != predict_year:
            plt.scatter(year, price, c='skyblue')
        else:
            plt.scatter(year, price, c='red', marker='^')

    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    #plt.show()
    return img



if __name__=='__main__':
    dongList = {'0': '개포동', '1': '논현동', '2': '대치동', '3': '도곡동', '4': '삼성동', '5': '세곡동',
                '6': '수서동','7': '신사동', '8': '압구정동', '9': '역삼동', '10': '일원동', '11': '청담동'}

    predict_year = int(input("예측할 년도를 입력하세요: "))
    print(dongList)
    predict_dong = input("예측할 동을 입력하세요: ").strip()
    predict_dong = dongList[predict_dong]
    apt_predict(predict_year, predict_dong)