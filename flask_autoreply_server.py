from flask import Flask, request, jsonify
import json
from get_response import run_demo

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['post'])
def auto_reply():
    if not request.data:
        return 'fail'
    con = request.json
    context = con['context']
    intent = con['intent']
    is_kb = con['is_kb']
    kb, response = run_demo(context, intent, is_kb)
    return jsonify({'kb': kb, 'response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)



