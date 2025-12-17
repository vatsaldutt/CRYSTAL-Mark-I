#          * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#         *                                                     *
#        *                    Crystal 2.5                      *
#       *        Crystal has the same futuristic theme        *
#      *              And a more improved Engine             *
#     *                                                     *
#    * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   
from selenium.webdriver.chrome.service import Service
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from nltk.stem.lancaster import LancasterStemmer
from subprocess import getstatusoutput as cmd
from pynput.keyboard import Key,Controller
from googletrans import Translator
from PyQt5.QtPrintSupport import * 
from Desktop import Ui_MainWindow
import speech_recognition as sr
from PyQt5 import QtCore, QtGui
from selenium import webdriver
from PyQt5.QtWidgets import *
from bs4 import BeautifulSoup
from mutagen.mp3 import MP3
from statistics import mode
from PyQt5.QtCore import * 
from secret import *
from PyQt5.QtGui import *
import face_recognition
from gtts import gTTS
import numpy as np
import tensorflow
import threading
import requests
import datetime
import warnings
import tflearn
import openai
import pickle
import urllib
import random
import json
import time
import nltk
import dlib
import cv2
import sys
import os

# nltk.download('punkt')

pixmap = ""
response = ""
data_list = []
first_time = True
first_time_weather = True
eye_contact_list = []
keyboard = Controller()
cap = cv2.VideoCapture(0)
openai.api_key = API_OPENAI
stemmer = LancasterStemmer()
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
language = open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/lang.txt", 'r').read()
predictor = dlib.shape_predictor("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/Face Recognition/shape_predictor_68_face_landmarks.dat")

op = webdriver.ChromeOptions()
op.add_argument('headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=op, service=Service('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/chromedrivers/chromedriver'))

warnings.filterwarnings('ignore')


def go_forward():
    pass


def go_backward():
    pass


def eye_contact(frame):
    frame = frame
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
            # print('Eye mode if statement failure')
            pass
    else:
        del eye_contact_list[0]
        try:
            eye_contact_list.append(convert[data])
        except:
            # print('Eye mode else statement failure')
            pass
        new_data = mode(eye_contact_list)
        for key, value in convert.items():
            if value == new_data:
                return key


def listen():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source)
            print("Recognizing...")
        dt = ''
        try:
            dt = r.recognize_google(audio, language='en-US')
            dt = f"You said: {dt}"
            print(dt)

        except sr.UnknownValueError:
            pass

        except sr.RequestError:
            print("Request results from Google Speech Recognition service error")

        with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt", 'w') as recognition:
            recognition.write( dt.replace("You said: ", ''))
        # return dt.replace("You said: ", '')

def speak(text):
    tts = gTTS(text)
    print(text)
    tts.save('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/audio.mp3')
    os.system('mpg123 /Users/vatsal/Desktop/CrystalOS/Crystal3.0/audio.mp3')
    # audio = MP3("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/audio.mp3")
    # audio_length = audio.info.length
    # time.sleep(int(audio_length)+3)
    time.sleep(3)
    with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt", 'w') as clearing:
        clearing.write('')


def get_weather_data():
    while True:
        global data_list
        global weather_data
        global first_time_weather
        global pixmap
        driver.get('https://www.google.com/search?q=weather')
        weather_data = driver.find_element_by_class_name('UQt4rd')
        weather_data = weather_data.text
        print('Whooooaaaaaah')
        print('Just passed through weather data')
        data_list = weather_data.split('\n')
        data_list[0] = data_list[0][0:-2]
        data_list.append(driver.find_element_by_id('wob_dc').text)
        weather_icon_link = driver.find_element_by_id('wob_tci').get_attribute('src')
        url = weather_icon_link
        with urllib.request.urlopen(url) as url1:
            weather_data = url1.read()
        pixmap = QPixmap()
        pixmap.loadFromData(weather_data)
        delay = 350
        if first_time_weather == True:
            temp = data_list[0]
            weather_details = (f'{data_list[1]}\
                {data_list[2]}\
                {data_list[3]}')
            weather_name = data_list[-1]
            print(f'Weather in your area is:\n, {temp}, {weather_details}, {weather_name}')
            first_time_weather = False
        else:
            time.sleep(delay)
        print('Updating Weather Information')
    

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

    for text in list1:
        list_text = text.split()
        if len(list_text) != 0:
            if list_text[-1] == 'Wikipedia':
                return 'According to the Wikipedia, '+str('/'.join(text.split()[0:-1]).replace('/', ' '))


def translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src=language, dest='en').text
    return translate_text


def reverse_translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src='en', dest=language).text
    return translate_text


def recognize_user(frame):
    img = frame
    path = '/Users/vatsal/Desktop/CrystalOS/Crystal3.0/img/face_recognition'
    images = []
    classNames = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    name = ''

    with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/models/face_rec', 'rb') as file:
        encodeListKnown = pickle.load(file)
    print('INITIATING FACIAL RECOGNITION PROTOCOL')
    print('SCANNING FACES FOR FACE RECOGNITION')
    while name == '':
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
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (205, 154, 79))
                cv2.putText(img, 'UNKNOWN', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    print('FACE RECOGNIZED')
    print('Welcome', name)
    
    return name


def wish_me(user):
    hr = int(datetime.datetime.now().hour)
    if 0 < hr < 12:
        wish = ("Good Morning " + user.lower().title() + "!")

    elif 12 <= hr <= 16:
        wish = ("Good Afternoon " + user.lower().title() + "!")

    else:
        wish = ("Good Evening " + user.lower().title() + "!")
    
    speak(wish)


def listen_continuously():
    while True:
        listen()


def wake_word(command):
    wake_words = ['crystal', 'kristol']
    command = command.lower()

    for wake_word in wake_words:
        if wake_word in command:
            return True

    return False


with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/intents.json") as file:
    data = json.load(file)

try:
    with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    print('TENSORFLOW MODEL FOUND')
    print('INITIATING MODEL...')
except FileNotFoundError:
    print('TENSORFLOW MODEL NOT FOUND')
    print('INITIATING TRAINING PROTOCOL...')
    os.system('/usr/local/bin/python3 /Users/vatsal/Desktop/CrystalOS/Crystal3.0/train.py')
    print('MODEL TRAINING PROTOCOL SUCCESS')
    with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/models/model.tflearn")
    print('MODEL READY TO BE USED')
except:
    model.fit(training, output, n_epoch=800, batch_size=8, show_metric=True)
    model.save("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/models/model.tflearn")
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


# class MainThread(QThread):
#     def __init__(self):
#         super(MainThread, self).__init__()

def run():
    print('EXECUTING')
    while True:
        os.system('clear')
        print('LOOPING')
        global response
        global first_time
        with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt', 'r') as recognition_text:
            command = recognition_text.read()
        command = command.lower()
        _, frame = cap.read()

        eyes = eye_mode(frame)
        print(eyes)
        if first_time == True:
            print('CAMERA FULLY INTIATED')
            wish_me(recognize_user(frame))
            first_time = False
        print(wake_word(command))
        if (wake_word(command)is True or eyes == "center") and command != "":
            command = command.replace('crystal ', '').replace(' crystal', '').replace('crystal', '')
            response = ""
            print('ENTERED INTO PROCESSING STAGE')
            results = model.predict([bag_of_words(command, words)])[0]
            results_index = np.argmax(results)
            tag = labels[int(results_index)]

            if tag == "volume up":
                for i in range(5):
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up) 

            elif tag == "volume down":
                for i in range(5):
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)

            elif tag == "brightness down":
                current_brightness-=10
                # sbc.set_brightness(current_brightness)

            elif tag == "brightness up":
                current_brightness+=10
                # sbc.set_brightness(current_brightness)

            # elif tag == "play/pause":
            #     keyboard.press(Key.media_play_pause)
            #     keyboard.release(Key.media_play_pause)

            elif tag == "forward":
                go_forward()

            elif tag == "backward":
                go_backward()              

            elif tag == "mute":
                print('Muting the sound...')
                keyboard.press(Key.media_volume_mute)
                keyboard.release(Key.media_volume_mute)
            
            elif command == "":
                response = response + ""
            
            with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/prompts.txt', 'r') as prompt:
                prompt_list = str(prompt.read()).split('||[end]||\n')
            with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/completions.txt', 'r') as completions:
                completions_list = str(completions.read()).split('||[end]||\n')
            if command.lower() in prompt_list:
                response = completions_list[prompt_list.index(command.lower())]
                print(response)
            if response == "" or response == None:
                try:
                    speak('Webscraping...')
                    response = web_scrapper(command)
                    print(response)
                except Exception as e:
                    speak('Error while webscraping: e')

            if response == "" or response == None:
                # confirmation = input('Enter Yes or no: ')
                confirmation = 'yes'
                with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/User.txt', 'r') as user_input:
                    usr_inp = user_input.read()
                with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/Crystal.txt', 'r') as crystal_input:
                    crstl_inp = crystal_input.read()
                if 'y' in confirmation.lower():
                    template = f"User:{usr_inp}\nCrystal:{crstl_inp}\nUser:{command}\nCrystal:"
                    print(template)
                    response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=template,
                        temperature=0.5,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0.6,
                        stop=["User:", " Crystal:"]
                    )
                    response = response.choices[0].text
                    with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/User.txt', 'w') as user_input:
                        user_input.write(str(command))
                    with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/Crystal.txt', 'w') as crystal_input:
                        crystal_input.write(response)
                else:
                    response = ""
            with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt", 'w') as clearing:
                clearing.write('')
            try:
                speak(response)
            except Exception as e:
                print("Error playing audio:\n", e)
        
        else:
            pass
                
        # cv2.imshow("Frame", frame)
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
        self.ui.movie = QtGui.QMovie("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/img/UI/Crystal new.gif")
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
            with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt', 'r') as speech:
                self.ui.textBrowser.setText(speech.read())
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

