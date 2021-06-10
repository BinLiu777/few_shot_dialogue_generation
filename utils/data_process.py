# import re
import regex as re
import json
from tqdm import tqdm

intents = ['返学费规则', '随单礼品问题', '绘本售前咨询', '金额查询', '返学费进度', '修改级别问题', '其他绘本问题', '倒计时查询', '返学费申请', 'VIP/退款申请']


def get_name(text):
    text = text.replace('~', '')
    patten = re.compile(r'\\n[\u4e00-\u9fa5]+：')
    res = re.findall(patten, text)
    res = [x.strip('\\n').strip('：') for x in res]
    res = set(res)
    res -= {'顾客'}
    return res.pop()


def clear_symbol(text):
    text = text.replace('[语音]', '')
    text = text.replace('[快捷回复]', '')
    text = text.replace('\\n', '')
    text = re.sub(r':[a-zA-Z]+:', '', text)  # 表情
    patten = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9\p{P}~]')
    res = re.findall(patten, text)
    res = ''.join(res)
    res = res.replace('*', '')
    res = res.replace('()', '')
    res = res.replace('(｡)', '')
    res = res.replace('(｡•︿•｡)', '')
    return res


def clear_spec_sent(text):
    text = re.sub(r'请您稍后对我的服务做出评价，谢谢，祝您生活愉快！', '', text)
    text = re.sub(r'请问还有((其他的)|(别的))*可以帮您的吗[\s\S]*没有的话请您稍后为我的服务做出评价。谢谢您', '', text)
    text = re.sub(r'您好，[\s\S]*是人工客服[\s\S]*工号[0-9]+', '', text)
    text = re.sub(r'((xw)|(您好))[，|！]+[\s\S]*有什么((可以)|(能够))*帮您(。|~|！|？|(呢、)|(呢？)|(的吗？))*', '', text)
    text = re.sub(r'嗨~[\s\S]*非常高兴为您服务！', '', text)
    text = re.sub(r'((Hi)|(您好))+[，|!]我是[\s\S]*很高兴再次遇到您(！)*', '', text)
    text = re.sub(r'(非常)*抱歉(哦)*，由于现在咨询人数较多，[\s\S]*正在逐一解答。', '', text)
    text = re.sub(r'请您不要着急，耐心等待下吧。小鱼仔很快就来！', '', text)
    text = re.sub(r'请您耐心等待下，不要着急。小鱼仔很快就来！', '', text)
    text = re.sub(r'非常抱歉，咨询的人有些多，不能及时回复您，请您耐心等待下，[\s\S]*马上赶来~', '', text)
    text = re.sub(r'请问还有其他(的问题)*可以帮您(的)?吗(？|~|(亲亲~))*', '', text)
    text = re.sub(r'如果我的服务可以帮助到您，麻烦您给我一个好评哦~感谢您的配合(!)*', '', text)
    text = re.sub(r'((您好，)|(嗨~))+如果我的服务有帮助到您，(麻烦您给我一个好评哦~)*感谢您的(支持)*(！)*', '', text)
    text = re.sub(r'(如果没有[，|~])*麻烦您给[我一]*个好评哦(~)*', '', text)
    text = re.sub(r'麻烦帮忙给个好评吧，谢谢[啦|亲]*！', '', text)
    text = re.sub(r'如果没有请您(稍后)*[为|对]我的服务(做|作)出评价，谢谢[！|。]', '', text)
    text = re.sub(r'为了保证服务质量，[\s\S]*结束了本次服务。', '', text)
    text = re.sub(r'如果您还有其他问题，请随时联系我哦！', '', text)
    text = re.sub(r'遇见您是我最大的幸运。祝您学习愉快，再见！', '', text)
    text = re.sub(r'如果我的服务可以帮助到您感谢您的配合！', '', text)
    text = re.sub(r'感谢您的配合！', '', text)
    text = re.sub(r'亲~客服系统目前无法识别语音哦，很抱歉给您带来不便。麻烦您以文字的形式说明问题吧，我会尽快为您解决！谢谢配合哦(~)*', '', text)
    text = re.sub(r'您好((, )|，)目前无法识别语音哦，很抱歉给您带来不便。麻烦您以文字的形式说明问题吧，我会尽快为您解决！谢谢配合~', '', text)
    text = re.sub(r'如果没有请您对我的服务做个评价哟~祝您学习愉快，再见~', '', text)
    text = re.sub(r'(亲|(您好))，看到您没有回复了那客服就先不打扰您了，如果有问题请随时联系客服((解决)|(处理))[呀|哦]！', '', text)
    return text


def clear_phone(text):
    text = re.sub(r'1[0-9]{10}', '电话号', text)
    return text


def clear_address(text):
    text = re.sub(r'image[\s\S]*.[png|jpg|jpeg]', '[image]', text)
    return text


def clear_number(text):
    text = re.sub(r'[0-9]+', '数字', text)
    return text


def process_data(lines):
    data = []
    raw_sents = []
    for i in tqdm(range(len(lines))):
        sess = {}
        sess['dialogue'] = []
        sess['scenario'] = {}

        saller_id, intent, text = lines[i].split('\t')

        text = clear_address(text)

        if '[image]' in text:
            continue
        # if intent not in ['返学费规则']:
        #     continue
        # if intent not in intents:
        #     continue

        raw_sents.append(text)
        text = clear_symbol(text)
        text = clear_number(text)
        text = clear_phone(text)
        text = clear_spec_sent(text)
        text = clear_symbol(text)   # clear_symbol必须前后处理两次

        patten = re.compile(r'(顾客：|{}：)'.format(saller_id))
        texts = re.split(patten, text)
        text_merge = []
        cur = ''
        for j in range(len(texts)):
            turn = texts[j]
            if turn:
                if turn == '顾客：' or turn == saller_id + '：':
                    try:
                        if texts[j+1] == '顾客：' or texts[j+1] == saller_id + '：' or texts[j+1] == '' or texts[j+1] == '。':
                            continue
                    except:
                        continue
                    if cur != turn:
                        text_merge.append(turn)
                        cur = turn
                    else:
                        continue
                else:
                    try:
                        text_merge[-1] += turn
                    except:
                        continue

        raw_sents.append(text_merge)

        for j in range(len(text_merge)):
            dialogue = {}
            turn = text_merge[j]
            name, utterance = turn.split('：', 1)
            dialogue['turn'] = name if name == '顾客' else '小鱼仔'
            dialogue['data'] = {}
            dialogue['data']['end_dialogue'] = True if j == len(text_merge) - 1 else False
            dialogue['data']['utterance'] = utterance
            dialogue['data']['requested'] = {}
            dialogue['data']['slots'] = {}
            sess['dialogue'].append(dialogue)

        sess['scenario']['kb'] = {}
        sess['scenario']['task'] = {}
        sess['scenario']['uuid'] = {}
        sess['scenario']['task']['intent'] = intent

        data.append(sess)

    json_str = json.dumps(raw_sents, indent=4, ensure_ascii=False)
    with open('customer_service/raw_sents.txt', 'w') as json_file:
        json_file.write(json_str)

    train_data = []
    dev_data = []
    test_data = []
    for i in range(len(data)):
        if i % 20 == 18:
            dev_data.append(data[i])
        elif i % 20 == 19:
            test_data.append(data[i])
        else:
            train_data.append(data[i])
    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))

    json_train = json.dumps(train_data, indent=4, ensure_ascii=False)
    with open('customer_service/customer_train.json', 'w') as json_file:
        json_file.write(json_train)

    json_dev = json.dumps(dev_data, indent=4, ensure_ascii=False)
    with open('customer_service/customer_dev.json', 'w') as json_file:
        json_file.write(json_dev)

    json_test = json.dumps(test_data, indent=4, ensure_ascii=False)
    with open('customer_service/customer_test.json', 'w') as json_file:
        json_file.write(json_test)


if __name__ == '__main__':

    with open('customer_service/intent_other_big.txt', 'r') as f:
        lines_ = f.readlines()
    lines1 = []
    for line in lines_:
        line = line.strip().split('\t')
        text = line[1]
        try:
            name = get_name(text)
        except:
            continue
        lines1.append('\t'.join([name, line[-1].strip('\n'), line[1]]))
    # process_data(lines1, 'customer_service/customer_service_2.json')

    with open('../data/customer_service/dialog.txt', 'r') as f:
        lines2 = f.readlines()
    # process_data(lines2, 'customer_service/customer_service1.json')

    lines = lines1 + lines2
    process_data(lines)


