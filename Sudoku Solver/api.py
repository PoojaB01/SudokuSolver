from flask import Flask, request, jsonify
from main import * 

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def hello_world():
    solveit(request.args['Query'])
    return "DONE"

if __name__ == '__main__':
    app.run()