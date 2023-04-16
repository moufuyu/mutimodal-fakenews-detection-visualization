# Processing Wikipedia Intorduction
def external_preprocess(introduction):
    intro_list = []
    for k, v in introduction.items():
        v = v.replace('"', '')
        v = v.replace('\\', '')
        v = v.replace('\n', '')
        v = v.replace("'", "")
        
        # 1文（ピリオドまでを抽出する）
        target = '.'
        idx = v.find(target)
        sentence = str()
        if idx != -1:
            sentence = v[:idx+1]
        else:
            sentence = v
        intro_list.append(sentence)
    return intro_list


def text_preprocessing(text):
    try:
        text = re.sub("\(.+?\)", "", text)
        text = re.sub("\(.+?\)", "", text)
        if text[-1] == ' ':
            text = text.rstrip()
        text = text.replace("colorized", "")
        text = text.replace("Colorized", "")
        text = text.replace("Colorized", "")
        text = text.replace("Colourised", "")
        text = text.replace("colourised", "")
        text = text.replace("PsBattle", "")
        text = text.replace("PsBattle:", "")
        text = text.replace("PsBattle: ", "")
        text = text.replace(':', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
    except Exception:
        pass
    
    return text

