import requests
import json

intent = ''
while True:
    intent = input('choose intent: 1: weather, 2: schedule, 3: navigate\n').strip()
    if intent == '1':
        intent = 'weather'
        break
    elif intent == '2':
        intent = 'schedule'
        break
    elif intent == '3':
        intent = 'navigate'
        break
    else:
        print('Wrong choice')
is_kb = True
while True:
    is_kb = input('whether use kb information: 1: yes, 2: no\n').strip()
    if is_kb == '1':
        is_kb = True
        # print('kb information:\n', kbs[intent], '\n')
        break
    elif is_kb == '2':
        is_kb = False
        break
    else:
        print('Wrong choice')
context = []

url = ' http://192.168.152.81:7777/'
headers = {'Content-Type': 'application/json'}
dio_turn = 0
while True:
    dio_turn += 1
    usr_utt = input('Input(say \'thank you\'to terminate the conversation): ')
    if 'thank you' in usr_utt or 'thanks' in usr_utt:
        print('bot: ', 'you are welcome!')
        break
    context.append(usr_utt)
    data = {'context': context, 'intent': intent, 'is_kb': is_kb}
    result = requests.post(url,  headers=headers, data=json.dumps(data)).json()
    response = result['response']
    kb = result['kb']
    print('bot:', response)
    if 'you are welcome' in response or 'you\' re welcome' in response:
        break
    context.append(response)


