#          * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#         *                                                     *
#        *                    Crystal 3.0                      *
#       *  Crystal has been fine tuned to be more Crystalish  *
#      *              With many more features!!!             *
#     *                                                     *
#    * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# Links!!!!
# wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
# http://raw.githubusercontent.com/davisking/dlib-models/master/dlib_face_recognition_resnet_model_v1.dat.bz2
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from nltk.stem.lancaster import LancasterStemmer
from selenium.webdriver.common.keys import Keys
from subprocess import getstatusoutput as cmd
from selenium.webdriver.common.by import By
from pynput.keyboard import Key,Controller
from playsound import playsound as ps
from googletrans import Translator
from PyQt5.QtPrintSupport import * 
from credentials import API_OPENAI
from Desktop import Ui_MainWindow
import speech_recognition as sr
from PyQt5 import QtCore, QtGui
from selenium import webdriver
from PyQt5.QtWidgets import *
from bs4 import BeautifulSoup
from statistics import mode
from PyQt5.QtCore import * 
from PyQt5.QtGui import *
import face_recognition
from csv import writer
from gtts import gTTS
from sys import exit
import pandas as pd
import numpy as np
import tensorflow
import threading
import requests
import datetime
import warnings
import imaplib
import tflearn
import openai
import pickle
import urllib
import random
import string
import email
import time
import json
import nltk
import dlib
import cv2
import sys
import os

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

path = cmd('pwd')[-1]

langs = {
    'afrikaans': 'af',
    'albanian': 'sq',
    'amharic': 'am',
    'arabic': 'ar',
    'armenian': 'hy',
    'azerbaijani': 'az',
    'basque': 'eu',
    'belarusian': 'be',
    'bengali': 'bn',
    'bosnian': 'bs',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'cebuano': 'ceb',
    'chichewa': 'ny',
    'chinese': 'zh-cn',
    'corsican': 'co',
    'croatian': 'hr',
    'czech': 'cs',
    'danish': 'da',
    'dutch': 'nl',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'filipino': 'tl',
    'finnish': 'fi',
    'french': 'fr',
    'frisian': 'fy',
    'galician': 'gl',
    'georgian': 'ka',
    'german': 'de',
    'greek': 'el',
    'gujarati': 'gu',
    'haitian creole': 'ht',
    'hausa': 'ha',
    'hawaiian': 'haw',
    'hebrew': 'he',
    'hindi': 'hi',
    'hmong': 'hmn',
    'hungarian': 'hu',
    'icelandic': 'is',
    'igbo': 'ig',
    'indonesian': 'id',
    'irish': 'ga',
    'italian': 'it',
    'japanese': 'ja',
    'javanese': 'jw',
    'kannada': 'kn',
    'kazakh': 'kk',
    'khmer': 'km',
    'korean': 'ko',
    'kurdish': 'ku',
    'kyrgyz': 'ky',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'lithuanian': 'lt',
    'luxembourgish': 'lb',
    'macedonian': 'mk',
    'malagasy': 'mg',
    'malay': 'ms',
    'malayalam': 'ml',
    'maltese': 'mt',
    'maori': 'mi',
    'marathi': 'mr',
    'mongolian': 'mn',
    'myanmar': 'my',
    'nepali': 'ne',
    'norwegian': 'no',
    'odia': 'or',
    'pashto': 'ps',
    'persian': 'fa',
    'polish': 'pl',
    'portuguese': 'pt',
    'punjabi': 'pa',
    'romanian': 'ro',
    'russian': 'ru',
    'samoan': 'sm',
    'scots gaelic': 'gd',
    'serbian': 'sr',
    'sesotho': 'st',
    'shona': 'sn',
    'sindhi': 'sd',
    'sinhala': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'somali': 'so',
    'spanish': 'es',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tajik': 'tg',
    'tamil': 'ta',
    'telugu': 'te',
    'thai': 'th',
    'turkish': 'tr',
    'turkmen': 'tk',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uyghur': 'ug',
    'uzbek': 'uz',
    'vietnamese': 'vi',
    'welsh': 'cy',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zulu': 'zu'
}

pixmap = ""
response = ""
data_list = []
first_time = True
eye_contact_list = []
keyboard = Controller()
first_time_notif = True
cap = cv2.VideoCapture(0)
first_time_weather = True
openai.api_key = API_OPENAI
stemmer = LancasterStemmer()
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
dontlisten = False
folder_location = "./"
predictor = dlib.shape_predictor(f"{folder_location}Face Recognition/shape_predictor_68_face_landmarks.dat")
with open(f'{folder_location}data/lang.txt', 'r') as data:
    language = data.read()

location, temp, weather_details, weather_name = "", "", "", ""


# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[32].id)
# engine.setProperty('rate', 200)

op = webdriver.ChromeOptions()
op.add_argument('headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=op, service=Service(f'{folder_location}chromedrivers/chromedriver'))
op.add_argument("user-agent=User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
driver1 = webdriver.Chrome('/Users/vatsal/Desktop/Crystal Operating System/Crystal Processing Core/Crystal3.0/chromedrivers/chromedriver', options=op)

warnings.filterwarnings('ignore')




with open(f'{folder_location}data/assistant_name.txt', 'r') as data:
    assistant_name = data.read()


def go_forward():
    pass


def go_backward():
    pass


def encode_file(file_name, level):
    ascii_letters = list(string.printable+'å∫ç∂´ƒ©˙ˆ')
    file = file_name
    file1 = open(file, 'r')
    data = file1.read()
    file1.close()
    file2 = open(file, 'w')
    for letter in data:
        file2.write(ascii_letters[ascii_letters.index(letter)+level])

def decode_file(file_name, level):
    encode_file(file_name, int('-'+str(level)))

def decoded_text(text, code):
    ascii_letters = list(string.printable+'å∫ç∂´ƒ©˙ˆ')
    clear_data = ""
    for letter in text:
        clear_data = clear_data + ascii_letters[ascii_letters.index(letter)-code]
    
    return clear_data


def voice_crystal(text):
    text = reverse_translate(text)
    tts = gTTS(text, lang=language)
    print(text)
    tts.save(f'{folder_location}audio.mp3')
    ps(f'{folder_location}audio.mp3')
    time.sleep(4)
    with open(f"{folder_location}data/recognition.txt", 'w') as recognition:
        recognition.write('')


# def voice_krish(text):
#     try:
#         engine.endLoop()
#     except Exception as e:
#         print(e)
#     engine.say(text)
#     engine.runAndWait()
#     print("Waiting")
#     time.sleep(4)
#     print("Wait complete")
#     with open(f"{folder_location}data/recognition.txt", 'w') as recognition:
#         recognition.write('')

# if assistant_name == 'krish':
#     speak = voice_krish
# else:
speak = voice_crystal


def eye_contact(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (150, 50, 0), 2)
        landmarks = predictor(gray, face)
        
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio == 0.0:
            return "closed"
        elif gaze_ratio <= 1:
            return "right"
        elif 1 < gaze_ratio < 2.7:
            return "center"
        else:
            return "left"


def get_gaze_ratio(eye_points, facial_landmarks, gray, frame):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    eye = cv2.resize(gray_eye, None, fx=6, fy=6)
    try:
        gaze_ratio = left_side_white/right_side_white
    except:
        gaze_ratio = 0
    
    return gaze_ratio

def eye_mode(frame):
    global eye_contact_list
    convert = {"right": 1, "center":2, "left": 3}
    data = (eye_contact(frame))
    if len(eye_contact_list) < 25:
        try:
            eye_contact_list.append(convert[data])
        except:
            pass
    else:
        del eye_contact_list[0]
        try:
            eye_contact_list.append(convert[data])
        except:
            pass
        new_data = mode(eye_contact_list)
        for key, value in convert.items():
            if value == new_data:
                return key


def listen():
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = r.listen(source)
        print("Recognizing...")
    dt = ''
    try:
        if language == 'en':
            listen_language = 'en-US'
        else:
            listen_language = language
        dt = r.recognize_google(audio, language=listen_language)
        dt = f"You said: {dt}"
        print(dt)

    except sr.UnknownValueError:
        return ""

    except sr.RequestError:
        print("Request results from Google Speech Recognition service error")
        return ""
    
    english = translate(dt.replace("You said: ", ''))
    with open(f"{folder_location}data/recognition.txt", 'w') as recognition:
            recognition.write(english)
    return english


def generate_response(me, other, query, contact):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f'''{contact}:{other}\nCrystal:{me}\n{contact}:{query}\nCrystal:''',
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[f" {contact}:", " Crystal:"]
        )
    print("Yous said:", query)
    print("I said:", response.choices[0].text)
    return response.choices[0].text

def check_notification():
    global first_time_notif
    notify = False
    sender = []
    subject = []
    try:
        driver1.get("https://web.whatsapp.com/")
        wait = WebDriverWait(driver1, 600)
        if first_time_notif == True:
            x_arg = '//div[@data-testid="qrcode"]'
            group_title = wait.until(EC.presence_of_element_located((
                By.XPATH, x_arg)))
            driver1.save_screenshot('qr.png')
            first_time_notif = False
        x_arg = '//span[@data-testid="filter"]'
        group_title = wait.until(EC.presence_of_element_located((
            By.XPATH, x_arg)))
        group_title.click()

        name = driver1.find_elements_by_class_name('ggj6brxn')
        names = []
        chats = []

        n = True
        for i in name:
            if n == True:
                names.append(i.text)
                n = False
            elif n == False:
                chats.append(i.text)
                n = True

        # Recieve unread emails
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        (retcode, capabilities) = mail.login('vatdut8994@gmail.com','zqaektslzdozfcgn')
        mail.list()
        mail.select('inbox')

        n=0
        (retcode, messages) = mail.search(None, '(UNSEEN)')
        if retcode == 'OK':
            for num in messages[0].split() :
                print('Processing ')
                n=n+1
                typ, data = mail.fetch(num,'(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        original = email.message_from_string(response_part[1].decode('utf-8'))

                        sender.append(original['From'])
                        subject.append(original['Subject'])
                        typ, data = mail.store(num,'+FLAGS','\Seen')
        if n > 0:
            for i in range(n):
                speak(f'You have a new email from {sender[i]} with the subject: {subject[i]}')
        else:
            print("No new E-mails")


        for i in name:
            i.click()
            try:
                current_contact = names[0]
                current_chat = chats[0]
                old_chat = driver1.find_elements_by_class_name('_1Gy50')
                for z in old_chat:
                    print(z.text)
                if current_chat != "":
                    speak("You have new messages from " + current_contact + " on WhatsApp. Would you like me to answer them?")
                    # dontlisten = True
                    shall_i = listen()
                    with open(f"{folder_location}data/recognition.txt", 'w') as recognition:
                        recognition.write('')
                    if 'yes' in shall_i.lower() or "yeah" in shall_i.lower():
                        inp_xpath = '//div[@class="p3_M1"]'
                        input_box = wait.until(EC.presence_of_element_located((
                            By.XPATH, inp_xpath)))
                        input_box.send_keys(generate_response(old_chat[-2].text, old_chat[-3].text, current_chat, current_contact) + Keys.ENTER)
                        del names[0]
                        del chats[0]
                    else:
                        speak(random.choice(["As you wish, Vatsal", "Okay Vatsal", "Okay Vatsal, I will leave them for you"]))
                    # dontlisten = False
                    notify = True
            except Exception as e:
                print(e)

        print(names)
        print(chats)
    except Exception as e:
        print(e)
    return notify

def get_weather_data():
    while True:
        global data_list
        global weather_data
        global first_time_weather
        global pixmap
        global location
        global temp
        global weather_details
        global weather_name
        global driver1
        global dontlisten
        print('Updating Screen Information')
        driver.get('https://www.google.com/search?q=weather')
        weather_data = driver.find_element_by_class_name('UQt4rd')
        weather_data = weather_data.text
        data_list = weather_data.split('\n')
        data_list[0] = data_list[0][0:-2]
        location = driver.find_element_by_id('wob_loc').text
        data_list.append(driver.find_element_by_id('wob_dc').text)
        weather_icon_link = driver.find_element_by_id('wob_tci').get_attribute('src')
        url = weather_icon_link
        with urllib.request.urlopen(url) as url1:
            weather_data = url1.read()
        pixmap = QPixmap()
        pixmap.loadFromData(weather_data)
        delay = 120
        check_notification()
        if first_time_weather == True:
            temp = data_list[0]
            weather_details = (f'{data_list[1]}\
                {data_list[2]}\
                {data_list[3]}')
            weather_name = data_list[-1]
            print(f'Weather in your {location} is:\n, {temp}, {weather_details}, {weather_name}')
            first_time_weather = False
        else:
            time.sleep(delay)
        print("Screen Information has been updated")

def tell_weather():
    global location
    global temp
    global weather_details
    global weather_name
    return f"weather today in {location} is:\n, {temp}, {', '.join(weather_details.split('                '))}, {weather_name}"
    

def web_scrapper(web_query):
    web_query = web_query.replace(' ', '%20').replace('=', '%3D').replace('+', "%2B")
    url = "https://www.google.com/search?q="+web_query
    driver.get(url)
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'html.parser')
    part_of_speeches = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition', 'conjunction', 'interjection', 'exclamation', 'numeral', 'article', 'determiner']

    list1 = []

    for i in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        for j in i.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            list1.append(j.text)
    
    try:
        return (soup.find('div', class_='BNeawe iBp4i AP7Wnd').text)
    except:
        pass
    try:
        element = driver.find_element_by_class_name("IZ6rdc")
        return (element.text)
    except:
        pass

    try:
        element = driver.find_element_by_class_name("Z0LcW CfV8xf")
        return (element.text)
    except:
        pass

    try:
        element = driver.find_element_by_class_name("ayqGOc kno-fb-ctx KBXm4e")
        return (element.text)
    except:
        pass

    if list1[0].split()[0] in part_of_speeches:
        if list1[0].split()[0][0] == "a":
            return 'As an '+list1[0].split()[0]+' it means '+list1[1]
        
        else:
            return 'As a '+list1[0].split()[0]+' it means '+list1[1]
    
    for text in list1:
        list_text = text.split()
        if len(list_text) != 0:
            if list_text[-1] == 'Wikipedia':
                return 'According to the Wikipedia, '+str('/'.join(text.split()[0:-1]).replace('/', ' '))
    
    answer_types = ['You would say that ', 'That would be ', "That's "]
    for i in soup.find_all('div'):
        for j in i.find_all('div'):
            for k in j.find_all('div'):
                for m in k.find_all('div'):
                    if 'MUxGbd u31kKd gsrt lyLwlc' in str(m):
                        translation = str(m.text).replace('Translation', '').replace('Translate', '')
    try:
        return random.choice(answer_types) + translation
    
    except:
        pass


def sadly_using_this(web_query):
    web_query = web_query.replace(' ', '%20').replace('=', '%3D').replace('+', "%2B")
    page_url = 'https://www.google.com/search?q=' + web_query
    source = requests.get(page_url).text
    soup = BeautifulSoup(source, 'html.parser')
    part_of_speeches = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition', 'conjunction', 'interjection', 'exclamation', 'numeral', 'article', 'determiner']

    list1 = []

    for i in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        for j in i.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            list1.append(j.text)
    
    try:
        return soup.find('div', class_='BNeawe iBp4i AP7Wnd').text
    except:
        pass

    if list1[0].split()[0] in part_of_speeches:
        if list1[0].split()[0][0] == "a":
            return 'As an '+list1[0].split()[0]+' it means '+list1[1]
        
        else:
            return 'As a '+list1[0].split()[0]+' it means '+list1[1]
    
    answer_types = ['You would say that ', 'That would be ', "That's "]
    for i in soup.find_all('div'):
        for j in i.find_all('div'):
            for k in j.find_all('div'):
                for m in k.find_all('div'):
                    if 'MUxGbd u31kKd gsrt lyLwlc' in str(m):
                        translation = str(m.text).replace('Translation', '').replace('Translate', '')
    try:
        return random.choice(answer_types) + translation
    
    except:
        pass
    
    try:
        driver.get(page_url)
        algebra_result = driver.find_elements_by_class_name('LPBLxf')
        return algebra_result[-1].text.split('\n')[-1]
    except:
        pass

    for text in list1:
        list_text = text.split()
        if len(list_text) != 0:
            if list_text[-1] == 'Wikipedia':
                return 'According to the Wikipedia, '+str('/'.join(text.split()[0:-1]).replace('/', ' '))
    urls = []
    for a in soup.find_all('a'):
        if a.get('href')[0:15] == '/url?q=https://':
            url = a.get('href').replace('/url?q=https://', '')
            urls.append(url[0: url.index('&sa')])
    
    for u in range(len(urls)):
        urls[u] = 'https://'+urls[u]


            
    if urls[0].split('/')[2] == "www.youtube.com":
        with open(f'{folder_location}data/youtube_query.txt', 'w') as youtube_query:
            youtube_query.write(web_query)
            youtube_query.close()
        os.system('python3f {folder_location}youtube.py')
        return "Here are some results from the web..."
    
    if "Duration" not in list1[0]:
        if len(list1[0].split()) > 10:
            try:
                return list1[0].split('...')[0].split("·")[1]
            
            except:
                return list1[0].split('...')[0]

    url_source = requests.get(urls[0]).text
    soup = BeautifulSoup(url_source, 'html.parser')
    url_text = []
    for i in soup.find_all("p"):
        url_text.append(i.text)

    paracount = 0
    for j in url_text:
        if len(j.split()) < 11:
            pass
        
        elif paracount == 1:
            return "According to the website "+urls[0].split('/')[2]+", "+j.split("\r\n\r")[0]

        else:
            paracount += 1


def translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src=language, dest='en').text
    return translate_text


def reverse_translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src='en', dest=language).text
    return translate_text


def recognize_user():
    path = f'{folder_location}img/face_recognition'
    name = ''
    images = []
    classNames = []
    myList = os.listdir(path)
    # name="Vatsal Dutt" 

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    with open(f'{folder_location}models/face_rec', 'rb') as file:
        encodeListKnown = pickle.load(file)
    print('INITIATING FACIAL RECOGNITION PROTOCOL')
    print('SCANNING FACES FOR FACE RECOGNITION')
    while name == '':
        _, img = cap.read()
        img = cv2.flip(img, 2)
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (205, 154, 79), 2)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            else:
                name = ""
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (205, 154, 79))
                cv2.putText(img, 'UNKNOWN', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                recognize_user()
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    print('FACE RECOGNIZED')
    print('Welcome', name)
    return name


def wish_me(user):
    hr = int(datetime.datetime.now().hour)
    if 0 < hr < 12:
        wish = ("Face Recognition Successful. Good Morning " + user.lower().title() + "!")

    elif 12 <= hr <= 16:
        wish = ("Face Recognition Successful. Good Afternoon " + user.lower().title() + "!")

    else:
        wish = ("Face Recognition Successful. Good Evening " + user.lower().title() + "!")
    
    speak(wish)


def listen_continuously():
    while True:
        global dontlisten
        if dontlisten:
            break
        listen()


def wake_word(command):
    global assistant_name
    wake_words_crystal = ['crystal', 'kristol', 'bristol']
    wake_words_krish = ['krish', 'krrish']
    command = command.lower()

    for wake_word in wake_words_crystal:
        if wake_word in command and assistant_name == 'crystal':
            return True, command.replace(wake_word, '')
    for wake_word in wake_words_krish:
        if wake_word in command and assistant_name == 'krish':
            return True, command.replace(wake_word, '')

    return False, command


with open(f"{folder_location}intents.json") as file:
    data = json.load(file)

try:
    with open(f"{folder_location}models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    print('TENSORFLOW MODEL FOUND')
    print('INITIATING MODEL...')
except FileNotFoundError:
    print('TENSORFLOW MODEL NOT FOUND')
    print('INITIATING TRAINING PROTOCOL...')
    os.system('/usr/local/bin/python3f {folder_location}train.py')
    print('MODEL TRAINING PROTOCOL SUCCESS')
    with open(f"{folder_location}models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load(f"{folder_location}models/model.tflearn")
    print('MODEL READY TO BE USED')
except:
    model.fit(training, output, n_epoch=800, batch_size=8, show_metric=True)
    model.save(f"{folder_location}models/model.tflearn")
    print('MODEL READY TO BE USED')


def bag_of_words(s, wrd):
    bag2 = [0 for _ in range(len(wrd))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(wrd):
            if w == se:
                bag2[i] = 1

    return np.array(bag2)


def run():
    global langs
    spoken = False
    print('EXECUTING')
    while True:
        # os.system('clear')
        global response
        global first_time
        global frame
        global language
        global speak
        global playing
        global assistant_name

        # if assistant_name == 'krish':
        #     speak = voice_krish
        # else:
        speak = voice_crystal


        with open(f'{folder_location}data/lang.txt', 'r') as data:
            language = data.read()
        with open(f'{folder_location}data/recognition.txt', 'r') as data:
            command = data.read()
        with open(f'{folder_location}data/User.txt', 'r') as data:
            usr_inp = data.read()
        with open(f'{folder_location}data/Crystal.txt', 'r') as data:
            crstl_inp = data.read()
        with open(f'{folder_location}data/assistant_name.txt', 'r') as data:
            assistant_name = data.read()
        raw_command = command
        try:
            if raw_command[-1] == " ":
                raw_command = raw_command[0:-1]
        except:
            raw_command = raw_command
        command = command.lower()
        _, frame = cap.read()
        try:
            eyes = eye_mode(frame)
        except:
            eyes = "closed"
        if first_time == True:
            username = recognize_user()
            username = username.title()
            if username.lower() == "vatsal dutt":
                wish_me(username)
            else:
                speak("Hello "+username+", running face recognition for Vatsal Dutt to access Crystal data")
            first_time = False
            username = username.split()[0]
        
        wake_word_recognized, command = wake_word(command)

        now = str(datetime.datetime.now()).split()[1].split(':')
        alarm = "10:23:PM".split(':')
        if alarm[-1] == "AM":
            hour = int(alarm[0])
        elif alarm[-1] == "PM":
            hour = int(alarm[0])+12
        
        if len(alarm) == 3:
            minute = int(alarm[1])
        elif len(alarm) == 2:
            minute = 0
        # print(now[0], hour, now[1], minute)
        if int(now[0]) == hour and int(now[1]) == minute and spoken == False:
            time.sleep(30)
            speak("Good Morning Mr. Dutt. The " + tell_weather())
            # for i in range(5):
            #     keyboard.press(Key.media_volume_up)
            #     keyboard.release(Key.media_volume_up)
            spoken = True

        if (wake_word_recognized is True or (eyes != "closed" and eyes != None and eyes != "None")) and command != "" and command != " ":
            command = command.replace(response.replace(',', '').replace('.', '').replace('!', '').replace('?', ''), "")
            print("Filtered command: ", command)
            response = ""
            light_on_list = ['light on', 'turn light on', 'initiate desk lamp protocol', 'turn on desk lamp', 'turn desk lamp on',
                            'turn on light', 'turn the light on', 'turn on the light', 'turn on the desk lamp', 'turn the desk lamp on',
                            'initiate the desk lamp protocol', 'desk lamp on', 'turn on my desk lamp', 'turn on my light']

            light_off_list = ['light off', 'turn light off', 'terminate desk lamp protocol', 'turn off desk lamp', 'turn desk lamp off',
                  "terminate the desk lamp protocol",'turn off light', 'turn the light off', 'turn off the light', 'turn off the desk lamp',
                   'turn the desk lamp off', 'desk lamp off', 'turn off my desk lamp', 'turn off my light']
            
            change_assistant_krish = ['i need to talk to krish', 'get krish', 'i need krish', 'change to krish', 'i want krish',
             'i would like to talk to krish', 'talk to krish', 'get me krish', 'give me krish', 'i want to talk to krish', 'i wanna talk to krish']

            change_assistant_crystal = ['i need to talk to crystal', 'get crystal', 'i need crystal', 'change to crystal', 'i want crystal',
             'i would like to talk to crystal', 'talk to crystal', 'get me crystal', 'give me crystal', 'i want to talk to crystal',
              'i wanna talk to crystal']
            
            notifications = ["do i have any notifications", "do i have any new notifications", "do i have any notification", "do i have any new notification",
            "check my notification", "check my notifications", "is there any new notification", "are there any new notification", "new notification", "new message",
            "do i have any messages", "do i have any new messages", "do i have any message", "do i have any new message","check my message",
            "check my messages", "is there any new message", "are there any new message", "new email",
            "do i have any emails", "do i have any new emails", "do i have any email", "do i have any new email","check my email",
            "check my emails", "is there any new email", "are there any new email"]

            print('ENTERED INTO PROCESSING STAGE')
            results = model.predict([bag_of_words(command, words)])[0]
            results_index = np.argmax(results)
            tag = labels[int(results_index)]
            if results[results_index] > 0.7:
                if tag == "volume up":
                    for i in range(5):
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up) 

                elif tag == "volume down":
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])
                    for i in range(5):
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)

                elif tag == "forward":
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])
                    go_forward()

                elif tag == "backward":
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])
                    go_backward()              

                elif tag == "mute":
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])
                    print('Muting the sound...')
                    keyboard.press(Key.media_volume_mute)
                    keyboard.release(Key.media_volume_mute)
            
            elif command == "" or command == " ":
                response = response + " "
            
            elif 'introduce yourself' in command:
                response = 'I am Crystal, an Artificial General Intelligence or AGI created by Vatsal Dutt to have consciousness and interact with humans like other humans. I can do almost anything a human mind can do, and some tasks beyond human intelligence.'
            
            for phrase in light_on_list:
                if phrase in command:
                    driver.get('http://192.168.1.69')
                    on = driver.find_element_by_id('off')
                    submit = driver.find_element_by_id('submit')
                    on.click()
                    submit.click()
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for phrase in light_off_list:
                if phrase in command:
                    driver.get('http://192.168.1.69')
                    off = driver.find_element_by_id('on')
                    submit = driver.find_element_by_id('submit')
                    off.click()
                    submit.click()
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for phrase in notifications:
                if phrase in command:
                    whattosay = check_notification()
                    if whattosay:
                        response = random.choice(['Okay', 'Sure!', "Command Accepted!"])
                    else:
                        response = "You currently do not have any new notifications"

            for phrase in change_assistant_crystal:
                if phrase in command:
                    with open(f'{folder_location}data/assistant_name.txt', 'w') as data:
                        data.write('crystal')
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for phrase in change_assistant_krish:
                if phrase in command:
                    with open(f'{folder_location}data/assistant_name.txt', 'w') as data:
                        data.write('krish')
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for language_ in langs:
                if ('speak' in command or 'talk' in command) and len(command.split()) < 4 and language_ in command:
                    print('CHANGING LANGUAGES')
                    with open(f'{folder_location}data/lang.txt', 'w') as data:
                        data.write(langs[language_])
                    language = langs[language_]
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            
            with open(f'{folder_location}data/prompts.txt', 'r') as prompt:
                prompt_list = decoded_text(str(prompt.read()), 8).split('||[end]||\n')
            with open(f'{folder_location}data/completions.txt', 'r') as completions:
                completions_list = decoded_text(str(completions.read()), 8).split('||[end]||\n')

            if command.lower() in prompt_list:
                response = completions_list[prompt_list.index(command.lower())]
                print(response)
            if response == "" or response == None and len(command.split()) > 1:
                try:
                    print("Trying to webscrape results")
                    response = web_scrapper(command)
                    print(response)
                except Exception as e:
                    print('Error while webscraping: e')

            if response == "" or response == None:
                template = ""
                file = pd.read_csv(f'{folder_location}/data/mentions.csv', index_col=False, delimiter='|')
                prmpts = file['prompt']
                cmpltns = file['completion']
                for c, prmpt in enumerate(prmpts, 0):
                    append_to_list = False
                    prmpt = prmpt.lower()
                    sentences = nltk.sent_tokenize(raw_command)
                    nouns = []

                    for sentence in sentences:
                        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
                            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                                nouns.append(word)
                    
                    for noun in nouns:
                        if noun in prmpt:
                            append_to_list = True
                            break
                        else:
                            pass
                    
                    if append_to_list == True:
                        template = template + f"{username}:{prmpt}\n{assistant_name}:{cmpltns[c]}\n"


                template = template + f"{username}:{usr_inp}\n{assistant_name}:{crstl_inp}\n{username}:{command}\n{assistant_name}:"
                confirmation = 'yes'
                if 'y' in confirmation.lower():
                    print(template)
                    response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=template,
                        temperature=0.4,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0.6,
                        stop=[f"{username}:", f" {assistant_name}:"]
                    )
                    response = response.choices[0].text
                    with open(f'{folder_location}data/Crystal.txt', 'w') as data:
                        data.write(response)
                    with open(f'{folder_location}data/User.txt', 'w') as data:
                        data.write(command)

                    data_to_append = [str(command+'|'), str(str(response).replace('\n', ''))]
                    if command not in prmpts:
                        with open(f'{folder_location}/data/mentions.csv', 'a') as file:
                            writer_object = writer(file)
                            writer_object.writerow(data_to_append)
                            file.close()
                else:
                    response = ""
            sorry = response.lower()
            if "sorry" in sorry and ("i can't" in sorry or "i don't" in sorry):
                response = sadly_using_this(command)
            
            if assistant_name == 'krish':
                response = response + random.choice([' Sir', ''])
            with open(f'{folder_location}data/recognition.txt', 'w') as data:
                data.write('')
            try:
                if assistant_name == 'krish':
                    speak_it = threading.Thread(target=speak, args=(response, ))
                    speak_it.start()
                else:
                    speak(response)
            except Exception as e:
                print("Error playing audio:\n", e)
        
        else:
            pass
                
        key = cv2.waitKey(1)
        if key == 27:
            break

processing = threading.Thread(target=run)
processing.start()
always_listen = threading.Thread(target=listen_continuously)
always_listen.start()
update_screen_data = threading.Thread(target=get_weather_data)
update_screen_data.start()


class Main(QMainWindow):
    global response
    global gif_path
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.startTask()
        self.ui.label_6.installEventFilter(self)
        self.ui.terminalCommand.returnPressed.connect(self.runCommand)

    def runCommand(self):
        command_line = self.ui.terminalCommand.text()
        p = cmd(command_line)[-1]
        self.ui.terminalOutput.clear()
        output = p
        self.ui.terminalOutput.setText(output)
        self.ui.terminalCommand.clear()
    
    def startTask(self):
        self.ui.movie = QtGui.QMovie(f'{folder_location}img/UI/Crystal new.gif')
        self.ui.label_2.setMovie(self.ui.movie)
        self.ui.movie.start()
        timer = QTimer(self)
        timer.timeout.connect(self.showData)
        timer.start(1000)
    
    def showData(self):
        global response
        global pixmap
        try:
            time_ = QTime.currentTime()
            date = QDate.currentDate()
            day = datetime.datetime.today().strftime('%A')
            hour_ = time_.toString('hh')
            minute_ = time_.toString('mm')
            meredian = 'AM'
            if int(hour_) > 12:
                hour_ = str(int(hour_)-12)
                meredian = 'PM'
            label_time = str(hour_)+':'+str(minute_)+meredian
            label_date = date.toString(Qt.ISODate)
            with open(f'{folder_location}data/recognition.txt', 'r') as data:
                speech = data.read()
                self.ui.textBrowser.setText(speech)
            self.ui.textBrowser_2.setText(response.replace('\n', ' '))
            self.ui.textBrowser_3.setText(label_time)
            self.ui.textBrowser_4.setText(label_date)
            self.ui.textBrowser_5.setText(day)
            try:
                self.ui.weathericon.setPixmap(pixmap)
            except Exception as e:
                print("Error presenting weather pixmap:\n", e)
            self.ui.temp.setText(data_list[0])
            self.ui.weatherdetails.setText(f'{data_list[1]}\
                {data_list[2]}\
                {data_list[3]}')
            self.ui.weathername.setText(data_list[-1])
            self.ui.weatherdetails.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.textBrowser_5.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.textBrowser_4.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.textBrowser_3.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.textBrowser_2.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.textBrowser.setAlignment(QtCore.Qt.AlignCenter)

        except Exception as e:
            print("Error while trying to show data:\n", e)

app = QApplication(sys.argv)
myos = Main()
myos.show()
exit(app.exec_())