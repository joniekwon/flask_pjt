import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO      #matplot img를 binary로 바꾸기 위해
from flask import Flask, request, render_template, send_file #(send_file : 이미지를 파일로 내보내기)
from _modules import gangnam_apt_module as lr
from _modules import fortune as lf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projectList():
    return render_template('/projectList.html')

@app.route('/projects/gangnam-apt')
def gangnam_apt():
    return render_template('/projects/gangnam_apt/main.html')

@app.route('/projects/gpt-fortune')
def gpt_fortune():
    return render_template('/projects/llm_pjt/fortune.html')

@app.route('/projects/gpt-fortune-result')
def gpt_fortune_chat():
    user_name = request.args.get('user_name')
    year = request.args.get('birth_year')
    month = request.args.get('birth_month')
    day = request.args.get('birth_day')
    query = request.args.get('user_query')
    try:
        answer = lf.connect_api(user_name, year, month, day, query)
        return render_template('/projects/llm_pjt/fortune_result.html', user_name=user_name, query=query, answer=answer)
    except:
        return render_template('/projects/llm_pjt/fortune_exceed.html')

@app.route('/projects/llm-project2')
def llm_project2():
    return render_template('/projects/llm_pjt/pjt2.html')

@app.route('/projects/check-fake')
def check_fake():
    return render_template('/projects/check_fake/main.html')

@app.route('/projects/gangnam-apt/predict', methods=['get'])
def project1Predict():
    predict_year = int(request.args.get('predict_year'))
    predict_dong = request.args.get('predict_dong')
    predict_prices, predict_dong, score = lr.apt_predict(predict_year, predict_dong)
    price1 = predict_prices[0]
    price2 = predict_prices[1]
    price3 = predict_prices[2]
    price4 = predict_prices[3]

    #print(predict_prices, score)
    plt.clf()
    return render_template('projects/gangnam_apt/result1.html', predict_year=predict_year,
                           predict_dong=predict_dong, predict_price=predict_prices[4], score=score,
                           price1=price1, price2=price2, price3=price3, price4=price4)

@app.route('/saveFig/<predict_year>/<predict_price>/<price1>/<price2>/<price3>/<price4>')
def saveFig(predict_year, predict_price, price1, price2, price3, price4):
    predict_year = int(predict_year)
    price1 = int(price1)
    price2 = int(price2)
    price3 = int(price3)
    price4 = int(price4)
    predict_prices = [price1, price2, price3, price4,int(predict_price)]
    colors = ['skyblue'] * 4 + ['pink']
    img = BytesIO()
    # # 한글 폰트 설정
    # plt.rcParams['font.family'] = 'Malgun Gothic'
    # plt.rcParams['font.size'] = 10

    years = [x for x in range(predict_year - 4, predict_year + 1)]

    plt.bar(years, predict_prices, color=colors)

    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    #img = lr.save_fig()
    return send_file(img, mimetype='image/png')


if __name__ =='__main__':
    app.run(debug=True)

