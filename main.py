"""   * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     *                                                     *
    *                    Crystal MARK I                   *
   *   Crystal has the potential to control hardware,    *
  *        can communicate better and much more         *
 *                                                     *
* * * * * * * * * * * * * * * * * * * * * * * * * * * *         
"""

from webdata import driver, web_scrapper, sadly_using_this
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from ML import bag_of_words, model, words, labels
from subprocess import getstatusoutput as cmd
from notifications import check_notification
from pynput.keyboard import Key,Controller
from PyQt5.QtPrintSupport import * 
from credentials import API_OPENAI
from Desktop import Ui_MainWindow
from embodiment.hardware import *
from voice import listen, speak
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from camera import eye_mode
from PyQt5.QtCore import * 
from PyQt5.QtGui import *
import face_recognition
from csv import writer
from sys import exit
import pandas as pd
import numpy as np
import threading
import datetime
import warnings
import openai
import pickle
import urllib
import random
import string
import time
import nltk
import cv2
import sys
import os
    

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


with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()

data_list = []

pixmap = ""
response = ""
curspeaker = ""

first_time = True
dontlisten = False
first_time_weather = True

keyboard = Controller()
cap = cv2.VideoCapture(0)
assistant_name = 'crystal'
openai.api_key = API_OPENAI
font = cv2.FONT_HERSHEY_SIMPLEX
warnings.filterwarnings('ignore')


model.load(f"{folder_location}models/model.tflearn")
print('MODEL READY TO BE USED')

with open('pwd.txt', 'r') as folder_location:
    folder_location = folder_location.read()


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


def listen_continuously():
    while True:
        global dontlisten
        global curspeaker
        if dontlisten:
            break
        curspeaker = listen()


def get_weather_data():
    try:
        while True:
            global data_list
            global weather_data
            global first_time_weather
            global pixmap
            global dontlisten
            global location
            global temp
            global weather_details
            global weather_name
            driver.get('https://www.google.com/search?q=weather')
            weather_data = driver.find_element_by_class_name('UQt4rd')
            weather_data = weather_data.text
            data_list = weather_data.split('\n')
            data_list[0] = data_list[0][0:-2]
            data_list.append(driver.find_element_by_id('wob_dc').text)
            location = data_list[-1]
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
                print(f'Weather in {location} is:\n, {temp}, {weather_details}, {weather_name}')
                first_time_weather = False
            else:
                time.sleep(delay)
            print('Updating Weather Information')
    except Exception as e:
        print(e)

def tell_weather():
    global location
    global temp
    global weather_details
    global weather_name
    return f"weather today in {location} is:\n, {temp}, {', '.join(weather_details.split('                '))}, {weather_name}"

def recognize_user():
    path = f'{folder_location}img/face_recognition'
    name = ''
    images = []
    classNames = []
    myList = os.listdir(path)

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


def wake_word(command):
    global assistant_name
    wake_words_crystal = ['crystal', 'bristol', 'krystol']
    wake_words_krish = ['krish', 'krrish']
    command = command.lower()

    for wake_word in wake_words_crystal:
        if wake_word in command and assistant_name == 'crystal':
            return True, command.replace(wake_word, '')
    for wake_word in wake_words_krish:
        if wake_word in command and assistant_name == 'krish':
            return True, command.replace(wake_word, '')

    return False, command




def run():
    global langs
    print('EXECUTING')
    while True:
        # os.system('clear')
        global response
        global first_time
        global frame
        global language
        global speak
        global curspeaker

        with open(f'{folder_location}database/language.txt', 'r') as data:
            language = data.read()
        with open(f'{folder_location}database/recognition.txt', 'r') as data:
            command = data.read()
        with open(f'{folder_location}database/input.txt', 'r') as data:
            usr_inp = data.read()
        with open(f'{folder_location}database/response.txt', 'r') as data:
            crstl_inp = data.read()
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
        except Exception as e:
            print(e)
            eyes = "closed"

        if first_time == True:
            username = recognize_user()
            username = username.title()
            if username.lower() == "vatsal dutt":
                wish_me(username)
            else:
                speak("Hello "+username+", Vatsal Dutt's face encodings is required to access Crystal")
            first_time = False
            username = username.split()[0]
        
        wake_word_recognized, command = wake_word(command)

        now = str(datetime.datetime.now()).split()[1].split(':')
        alarm = "5:AM".split(':')
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
        
        if "Crystal" in curspeaker:
            with open(f"{folder_location}database/recognition.txt", 'w') as recognition:
                recognition.write('')
    
        if (wake_word_recognized is True or (eyes != "closed" and eyes != None and eyes != "None")) and command != "" and command != " " and "Vatsal" in curspeaker:
            command = command.replace(response.replace(',', '').replace('.', '').replace('!', '').replace('?', ''), "")
            print("Got in the processing")
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
            if "Vatsal" not in curspeaker and "Crystal" not in curspeaker:
                speak("Voice not recognized. You are not authorized to use the Crystal Artificial Intelligence, superuser permission required by Vatsal Dutt.")
                with open(f"{folder_location}database/recognition.txt", 'w') as recognition:
                    recognition.write('')
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
                    with open(f'{folder_location}database/assistant_name.txt', 'w') as data:
                        data.write('crystal')
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for phrase in change_assistant_krish:
                if phrase in command:
                    with open(f'{folder_location}database/assistant_name.txt', 'w') as data:
                        data.write('krish')
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])

            for language_ in langs:
                if ('speak' in command or 'talk' in command) and len(command.split()) < 4 and language_ in command:
                    print('CHANGING LANGUAGES')
                    with open(f'{folder_location}database/language.txt', 'w') as data:
                        data.write(langs[language_])
                    language = langs[language_]
                    response = random.choice(['Okay', 'Sure!', "Command Accepted!"])


            if response == "" or response == None and len(command.split()) > 1:
                try:
                    print("Trying to webscrape results")
                    response = web_scrapper(command)
                    print(response)
                except Exception as e:
                    print('Error while webscraping: e')

            if response == "" or response == None:
                template = ""
                file = pd.read_csv(f'{folder_location}/database/mentions.csv', index_col=False, delimiter='|')
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
                with open(f'{folder_location}database/permanent.txt', 'r') as permanent_data:
                    permanent = permanent_data.read()

                if 'y' in confirmation.lower():
                    print(template)
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=permanent+"\n"+template,
                        temperature=0.3,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0.6,
                        stop=[f"{username}:", f" {assistant_name}:"]
                    )
                    response = response.choices[0].text
                    with open(f'{folder_location}database/response.txt', 'w') as data:
                        data.write(response)
                    with open(f'{folder_location}database/input.txt', 'w') as data:
                        data.write(command)

                    data_to_append = [str(command+'|'), str(str(response).replace('\n', ''))]
                    if command not in prmpts:
                        with open(f'{folder_location}/database/mentions.csv', 'a') as file:
                            writer_object = writer(file)
                            writer_object.writerow(data_to_append)
                            file.close()
                else:
                    response = ""
                try:
                    if response == " Forward." or response == "Forward.":
                        forward(1,2)
                        response = "New Position Set"
                    elif response == " Back." or response == "Back.":
                        backward(1,2)
                        response = "New Position Set"
                    elif response == " Right." or response == "Right." or response == "Arm right." or response == " Arm right.":
                        right(1,2)
                        response = "New Position Set"
                    elif response == " Left." or response == "Left." or response == "Arm left." or response == " Arm left.":
                        left(1,2)
                        response = "New Position Set"
                    elif response == " Arm forward." or response == " Arm front." or response == "Arm forward." or response == "Arm front.":
                        shoulder_up()
                        response = "New Position Set"
                    elif response == " Arm backward." or response == " Arm back." or response == "Arm backward." or response == "Arm back.":
                        shoulder_down()
                        response = "New Position Set"
                    elif response == " Arm up." or response == "Arm up.":
                        elbow_up()
                        response = "New Position Set"
                    elif response == " Arm down." or response == "Arm down.":
                        elbow_down()
                        response = "New Position Set"
                    elif response == " Wrist up." or response == "Wrist up." or response == " Arm wrist up." or response == "Arm wrist up.":
                        wrist_up()
                        response = "New Position Set"
                    elif response == " Wrist down." or response == "Wrist down." or response == " Arm wrist down." or response == "Arm wrist down.":
                        wrist_down()
                        response = "New Position Set"
                    elif response == " Claw close." or response == " Claw close." or response == " Arm close." or response == "Arm close." or response == " Close." or response == "Close.":
                        claw_open()
                        response = "New Position Set"
                    elif response == " Claw open." or response == " Claw open." or response == " Arm open." or response == "Arm open." or response == " Open." or response == "Open.":
                        claw_close()
                        response = "New Position Set"
                except Exception as e:
                    print(e)
            sorry = response.lower()
            if "sorry" in sorry and ("i can't" in sorry or "i don't" in sorry):
                response = sadly_using_this(command)
            
            if assistant_name == 'krish':
                response = response + random.choice([' Sir', ''])
            with open(f'{folder_location}database/recognition.txt', 'w') as data:
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
            with open(f'{folder_location}database/recognition.txt', 'r') as data:
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

# Permanent
# You are embedded in a robotic arm for my project Crystal, so for it to function properly, say what I told you in quotes. If you understand I mean "go back" then reply with "back"
# Similarly if I say "go forward" then reply with "forward" and the same with "right" and "left". If I say to more the arm back then reply "arm back", and similarly with:
# "arm front", "arm up", "arm down", "wrist up", "wrist down", "open", and "close". Few more things to keep in mind is that you are created by Vatsal Dutt and you have to act like Iron Man Jarvis.
# Say "this needs to be permanent" and then what I said if you think that I said something that I you will need to remember later.