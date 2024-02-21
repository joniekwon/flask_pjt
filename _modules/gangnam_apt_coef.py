import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 10
def data_preprocessing(fileName):
    # 데이터 전처리
    from sklearn.preprocessing import PolynomialFeatures        #특성을 만들거나 전처리를 위해 제공되는 클래스
    data = pd.read_csv(f"{fileName}", encoding='cp949', engine='python', index_col=False)

    # 행정 구역을 integer로 변환
    area = {'개포동':0,'논현동':1,'대치동':2,'도곡동':3,'삼성동':4,'세곡동':5,
            '수서동':6,'신사동':7,'압구정동':8,'역삼동':9,'일원동':10,'청담동':11}

    #print(data.columns)        #컬럼명 확인
    #'dongName', 'parkCnt', 'populationCnt', 'academyCnt',
    #       'aptName', 'area', 'contractYearMonth', 'contractDate', 'price',
    #       'floor', 'buildYear', 'PM2.5', 'ppm', 'childeCenterCnt', 'martDistance','medicalDistance']

    data['dongName'] = data['dongName'].map(area, na_action=None)       # 동 이름 변경
    data['price'] = data['price'].map(lambda x:str(x).replace(',', ''), na_action=None).astype('float64').apply(lambda x:x*3.3)
    data['aptName'] = pd.factorize(data['aptName'])[0] #1부터 시작하고 싶으면 [1]

    #print(data['aptName'])
    return data

def apt_predict(data):
    features = data.loc[:,data.columns!=('price')]
    #xValue = features.loc[:,features.columns!=('ppm')]
    #print(xValue)
    price = data['price'].to_numpy()
    #print(features)

    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(features, price, random_state=38)

    lr = LinearRegression()
    lr.fit(train_input, train_target)
    score = lr.score(test_input, test_target)

    result = pd.DataFrame({'features': features.columns, 'coef_:':lr.coef_.tolist()})
    result.to_csv("./data/output.csv", encoding='cp949')
    #
    print(f"lr.coef_: {lr.coef_}")
    print(f"lr.intercept_: {lr.intercept_}")
    print(f"score: {score}")

def apt_predict_poly(data):
    features = data.loc[:,data.columns!=('price')]
    #xValue = features.loc[:,features.columns!=('ppm')]
    #print(xValue)
    price = data['price'].to_numpy()
    #print(features)

    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(features, price, random_state=38)
    train_poly = np.column_stack((train_input ** 2, train_input))
    test_poly = np.column_stack((test_input ** 2, test_input))

    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    score = lr.score(test_poly, test_target)

    result = pd.DataFrame({'features': features.columns, 'coef_**2':lr.coef_[:len(features.columns)], 'coef_':lr.coef_[len(features.columns):]})
    result.to_csv("./data/output.csv", encoding='cp949')
    #
    print(f"lr.coef_: {lr.coef_}")
    print(f"lr.intercept_: {lr.intercept_}")
    print(f"score: {score}")

if __name__=='__main__':
    dongList = {'0': '개포동', '1': '논현동', '2': '대치동', '3': '도곡동', '4': '삼성동', '5': '세곡동',
                '6': '수서동','7': '신사동', '8': '압구정동', '9': '역삼동', '10': '일원동', '11': '청담동'}

    #predict_year = int(input("예측할 년도를 입력하세요: "))
    #print(dongList)
    #predict_dong = int(input("예측할 동을 입력하세요: ").strip())
    data = data_preprocessing("./data/output/allDf.csv")
    #predict_dong = '0'
    #predict_year = 2022
    #predict_dong = dongList[predict_dong]

    #apt_predict(data)
    apt_predict_poly(data)