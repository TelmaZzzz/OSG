import jieba
import jieba.posseg as pseg
import operator
import json
from collections import Counter
 
 
# Data structure for holding data
class Word():
    def __init__(self, char, freq = 0, deg = 0):
        self.freq = freq
        self.deg = deg
        self.char = char
 
    def returnScore(self):
        return self.deg/self.freq
 
    def updateOccur(self, phraseLength):
        self.freq += 1
        self.deg += phraseLength
 
    def getChar(self):
        return self.char
 
    def updateFreq(self):
        self.freq += 1
 
    def getFreq(self):
        return self.freq
 
# Check if contains num
def notNumStr(instr):
    for item in instr:
        if '\u0041' <= item <= '\u005a' or ('\u0061' <= item <='\u007a') or item.isdigit():
            return False
    return True
 
# Read Target Case if Json
def readSingleTestCases(testFile):
    with open(testFile) as json_data:
        try:
            testData = json.load(json_data)
        except:
            # This try block deals with incorrect json format that has ' instead of "
            data = json_data.read().replace("'",'"')
            try:
                testData = json.loads(data)
                # This try block deals with empty transcript file
            except:
                return ""
    returnString = ""
    for item in testData:
        try:
            returnString += item['text']
        except:
            returnString += item['statement']
    return returnString
 
def run(rawText):
    # Construct Stopword Lib
    swLibList = [line.rstrip('\n') for line in open("/users10/lyzhang/Datasets/stopwords/scu_stopwords.txt",'r',encoding='utf-8')]
    # Construct Phrase Deliminator Lib
    conjLibList = [line.rstrip('\n') for line in open("/users10/lyzhang/Datasets/stopwords/scu_stopwords.txt",'r',encoding='utf-8')]
 
    # Cut Text
    rawtextList = pseg.cut(rawText)
 
    # Construct List of Phrases and Preliminary textList
    textList = []
    listofSingleWord = dict()
    lastWord = ''
    poSPrty = ['m','x','uj','ul','mq','u','v','f']
    meaningfulCount = 0
    checklist = []
    for eachWord, flag in rawtextList:
        checklist.append([eachWord,flag])
        if eachWord in conjLibList or not notNumStr(eachWord) or eachWord in swLibList or flag in poSPrty or eachWord == '\n':
            if lastWord != '|':
                textList.append("|")
                lastWord = "|"
        elif eachWord not in swLibList and eachWord != '\n':
            textList.append(eachWord)
            meaningfulCount += 1
            if eachWord not in listofSingleWord:
                listofSingleWord[eachWord] = Word(eachWord)
            lastWord = ''
 
    # Construct List of list that has phrases as wrds
    newList = []
    tempList = []
    for everyWord in textList:
        if everyWord != '|':
            tempList.append(everyWord)
        else:
            newList.append(tempList)
            tempList = []
 
    tempStr = ''
    for everyWord in textList:
        if everyWord != '|':
            tempStr += everyWord + '|'
        else:
            if tempStr[:-1] not in listofSingleWord:
                listofSingleWord[tempStr[:-1]] = Word(tempStr[:-1])
                tempStr = ''
 
    # Update the entire List
    for everyPhrase in newList:
        res = ''
        for everyWord in everyPhrase:
            listofSingleWord[everyWord].updateOccur(len(everyPhrase))
            res += everyWord + '|'
        phraseKey = res[:-1]
        if phraseKey not in listofSingleWord:
            listofSingleWord[phraseKey] = Word(phraseKey)
        else:
            listofSingleWord[phraseKey].updateFreq()
 
    # Get score for entire Set
    outputList = dict()
    for everyPhrase in newList:
 
        # if len(everyPhrase) > 5:
        #     continue
        score = 0
        phraseString = ''
        outStr = ''
        for everyWord in everyPhrase:
            score += listofSingleWord[everyWord].returnScore()
            phraseString += everyWord + '|'
            outStr += everyWord
        phraseKey = phraseString[:-1]
        freq = listofSingleWord[phraseKey].getFreq()
        # if freq / meaningfulCount < 0.01 and freq < 3 :
        #     continue
        if len(outStr) <= 2:
            continue
        outputList[outStr] = score
 
    sorted_list = sorted(outputList.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_list[:16]
 
if __name__ == '__main__':
    sentence = "王羽熙有绘画天赋，3岁开始学画，很快就画得“有模有样”但他对数字极不敏感，上高一时，王羽熙在一次满分为150分的数学测验中仅得了3分：只答对了一道选择题。这么一个严重偏科的孩子却能两次进入北京顶尖中学－－人大附中，这多少让人称奇。但他自己认为，这全是因为画。王羽熙六年级那年的寒假，父母抱着试试看的心理给人大附中的刘彭芝校长打了一个电话。到了学校，刘校长让他现场画一幅画，他画了一幅老虎。因为这一次现场考核，王羽熙顺利地进入了人大附中。初中毕业时，因为理科成绩较差，王羽熙到了其他学校，但他不快乐。后来他妈妈拿着他的《新游记》的漫画书稿再一次找到了刘校长，刘校长认为像他这样的孩子更需要一个宽松、宽容的学习环境。就这样，王羽熙又回到了人大附中。一个偏科生能两次进入人大附中，确实挺幸运，但是只有幸运是不够的，还要有真本事。当别的孩子在玩的时候，王羽熙通常只做两件事：看《西游记》和画画。他对《西游记》的痴迷近乎狂热，他的第一套漫画书就是由《西游记》改编的。因为画画，他的手握笔的地方和与纸张经常接触的地方都有着厚厚的老茧。其实，高中三年对王羽熙的成长至关重要，因为在这三年里，他的收获颇多。高一下半学期，王羽熙开始在班里策划和执导英语剧《魔戒》，学期末，他们就在全校师生面前演出了一次。但王羽熙自己却认为那次准备得太仓促了，想重新准备再演一次。“重新演一次”，说起来容易但实际操作难度很大：要说服学校和老师，要说服学校再安排演出时间、再批礼堂。王羽熙带领同学最终搞定了所有事情。高二时，二次上演的《魔戒》获得了极大的成功。而《魔戒》的成功，也使王羽熙找到了电影美术这样一条更适合自己发展的道路。无论梦想有多么美好，高考仍然是王羽熙必须跨越的一个高台。虽然王羽熙认为，作为学生应该经历自己必须经历的考验，同时还要尽力做好。但对他这样一个偏科严重的孩子来说，高考前夕的压力还是可想而知的。那时，连一向支持他画画的妈妈也开始给他加压，不允许他画画了，让他把省出来的时间用来学数学。可见，当这个决定他命运的关键点出现时，父母承受的压力也到了极限。好在王羽熙不是一个轻言放弃的人。在他自己的努力下，在老师和同学的帮助下，王羽熙的高考数学成绩突破了个位数：33分。这对他来说已经是很不错的成绩了。而后王羽熙考上了大学，在大学里他的才华展现得淋漓尽致。3年前，大学还没毕业的他，参加了北京奥运会开幕式的部分动画制作；还是3年前，他作为唯一的学生代表参加了法国电影节。现在25岁的王羽熙，是一家广告公司的股东。他有一个自己的5年计划：用一两年的时间体验当老板的滋味，再之后的两三年里潜心增进绘画技艺，到他30岁的时候，进入迪斯尼那样世界顶尖的动漫公司。"
    print(sentence)
    result = run(sentence)
    print(result)