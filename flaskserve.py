from flask import Flask, request, render_template
from functions import log_reg, auto_reg, display, algo_comp
from datetime import date

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/options')
def options():
    return render_template('options.html')


@app.route('/data_display')
def data_display():
    data1 = display()
    return render_template('dataset.html', shape=data1[0], col=data1[1], head=data1[2], tail=data1[3],
                           desc=data1[4])


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return render_template('prediction.html')

    elif request.method == 'POST':
        add = "/static/autocln.png"
        day = int(request.form.get('day'))
        month = int(request.form.get('month'))
        year = int(request.form.get('year'))
        n1 = date.today()
        d0 = date(2014, 12, 31)
        d1 = date(n1.year, n1.month, n1.day)
        d2 = date(year, month, day)
        delta1 = (d1 - d0).days
        delta2 = (d2 - d0).days
        result = auto_reg(delta1, delta2)  # junk value
        len1 = len(result[0][0])
        print('len= %d and type of result= %s' % (len1, type(result)))
        return render_template('prediction_result.html', res=result[1], table=result[0], len=len1)


@app.route('/detection', methods=['POST', 'GET'])
def detection():
    if request.method == 'GET':
        return render_template('detection.html')

    elif request.method == 'POST':
        pm = float(request.form['pm'])
        dew_point = float(request.form['dew_point'])
        temperature = float(request.form['temperature'])
        pressure = float(request.form['pressure'])
        windspeed = float(request.form['windspeed'])
        prediction = log_reg(pm, dew_point, temperature, pressure, windspeed)
        if prediction == 0:
            prediction = "not polluted"
        else:
            prediction = " polluted"

        return render_template('detection_result.html', res=prediction)


if __name__ == '__main__':
    app.run(debug=True)
