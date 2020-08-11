
from flask import Flask, jsonify, request
from src.train_dkt import run

# flask框架提供能力评估接口
app = Flask(__name__)


@app.route('/')
def index():
    return 'hello world'


@app.route('/getAbility',methods = ['GET','POST'])
def getAbility():
    pid =int(request.args.get('pid'))

    try:
        ability = run([pid])
        result = {"status": "200", "data": ability}
        return jsonify(result)
    finally:
        ability = 0.5
        result = {"status": "200", "data": ability}
        return jsonify(result)
    return jsonify(pid)




if __name__ == '__main__':
    app.debug = True
    app.run(host = '127.0.0.1',port = 5000)
