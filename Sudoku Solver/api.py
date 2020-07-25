from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def hello_world():
    d={}
    d['query'] = str(request.args['Query'])
    return jsonify(d)

if __name__ == '__main__':
    app.run()