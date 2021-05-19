import re

def clean_text(s):

    text = re.sub('\(рис. \d+\)',"", s)
    text1 = re.sub('Глава \d+.',"", text)
    text2 = re.sub('[А-Я]{2,}',"", text1)
    text3 = re.sub('\(рис.\s\d+,\s\d+\)', "", text2)
    text4 = re.sub('\(рис.\s\d+\s-\s\d+\)', "", text3)
    text5 = re.sub('\(см.\sрис.\s\d+\)', "", text4)

    return text5

with open('Train_text.txt', 'r', encoding='cp1251') as target:
    text = target.read()
