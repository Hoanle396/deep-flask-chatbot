import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pickle
import json
from difflib import get_close_matches
import tflearn
import random
import nltk
import requests
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
from bs4 import BeautifulSoup


def loadInitNet():
    data = pickle.load(open("D:/Code/chatbot/src/models/training_data", "rb"))
    train_x = data["train_x"]
    train_y = data["train_y"]
    words = data["words"]
    classes = data["classes"]
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
    net = tflearn.regression(net)
    return net, words, classes


class ChatBot(object):
    def __init__(self):
        self.net, self.words, self.classes = loadInitNet()
        model = tflearn.DNN(self.net, tensorboard_dir="tflearn_logs")
        model.load("D:/Code/chatbot/src/models/main/model.tflearn")
        with open(
            "D:/Code/chatbot/src/models/data/data.json", encoding="utf8"
        ) as json_data:
            self.intents = json.load(json_data)
        self.model = model

    def chat(self, msg):
        msg = msg.lower()
        answer = self.response(msg)
        if answer is not None:
            return answer
        else:
            return self.query(msg)

    def classify(self, sentence):
        results = self.model.predict([self.bow(sentence, self.words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > 0.2]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        return return_list

    def response(self, sentence, userID="123", show_details=False):
        results = self.classify(sentence)
        if results:
            while results:
                for i in self.intents["intents"]:
                    if results[0][1] > 0.7:
                        if i["tag"] == results[0][0]:
                            return random.choice(i["responses"])
                results.pop(0)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=False):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def query(self, text):
        user_query = text
        URL = "https://www.google.co.in/search?q=" + user_query
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57"
        }
        page = requests.get(URL, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        bebe = str(soup)
        if 'class="LGOjhe"' in bebe:
            sult = soup.find(class_="LGOjhe").find(class_="hgKElc").get_text()
            return sult
        # dịch
        if 'class="Y2IQFc"' in bebe:
            sult = soup.find(id="tw-target-text").get_text()
            return sult
        elif 'class="vk_bk dDoNo FzvWSb"' in bebe:
            sult = soup.find(class_="vk_bk dDoNo FzvWSb").get_text()
            return sult
        # thông tin cơ bản
        elif 'class="Z0LcW"' in bebe:
            sult = soup.find(class_="Z0LcW").get_text()
            return sult
        # giá bicoin
        elif 'class="pclqee"' in bebe:
            sult = soup.find(class_="pclqee").get_text()
            return sult + " VNĐ"
        elif 'class="LGOjhe"' in bebe:
            sult = soup.find(class_="LGOjhe").find(class_="hgKElc").get_text()
            return sult
        # lyric
        elif 'class="FzvWSb"' in bebe:
            sult = soup.find(class_="FzvWSb").get_text()
            return sult
        # máy tính
        elif 'class="z7BZJb XSNERd"' in bebe:
            sult = soup.find(class_="qv3Wpe").get_text()
            return sult
        elif 'class="dDoNo ikb4Bb gsrt"' in bebe:
            sult = soup.find(class_="dDoNo ikb4Bb gsrt").get_text()
            return sult
        elif 'class="ayRjaf"' in bebe:
            sult = soup.find(class_="zCubwf").get_text()
            return sult
        # đổi các đơn vị
        elif 'class="dDoNo vrBOv vk_bk"' in bebe:
            sult = soup.find(class_="dDoNo vrBOv vk_bk").get_text()
            return sult
        # descript
        elif 'class="hgKElc"' in bebe:
            sult = soup.find(class_="hgKElc").get_text()
            return sult
        # thoi tiet
        elif 'class="UQt4rd"' in bebe:
            nhietdo = " Nhiệt độ: " + soup.find(class_="q8U8x").get_text() + "°C "
            doam = " Độ ẩm: " + soup.find(id="wob_hm").get_text()
            mua = " Khả năng có mưa: " + soup.find(id="wob_pp").get_text()
            gdp = soup.find(class_="wob_tci")
            wheather = gdp["alt"]
            nam = wheather + nhietdo + mua + doam
            return nam
        elif 'class="gsrt vk_bk FzvWSb YwPhnf"' in bebe:
            sult = soup.find(class_="gsrt vk_bk FzvWSb YwPhnf").get_text()
            return sult
        else:
            if len(text) > 0:
                return "Xin lỗi tôi không hiểu bạn nói gì."
