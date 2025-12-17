#          * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#         *                                                     *
#        *                    Crystal 2.0                      *
#       *   This version is for developing a better engine    *
#     *                                                     *
#    * * * * * * * * * * * * * * * * * * * * * * * * * * * *

from selenium.webdriver.chrome.service import Service
from pynput.keyboard import Key, Controller
import speech_recognition as sr
from selenium import webdriver
from bs4 import BeautifulSoup
from statistics import mode
from mutagen.mp3 import MP3
from gtts import gTTS
import numpy as np
import threading
import requests
import openai
import random
import time
import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
keyboard = Controller()
predictor = dlib.shape_predictor("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/Face Recognition/shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
eye_contact_list = []

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
    print(data)
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

op = webdriver.ChromeOptions()
op.add_argument('headless')
op.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=op, service=Service('/Users/vatsal/Desktop/CrystalOS/Crystal/chromedrivers/chromedriver'))


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
    os.system('open -a Music -g /Users/vatsal/Desktop/CrystalOS/Crystal3.0/audio.mp3')
    

openai.api_key = "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"

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
        print('Not in big bold text')

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

cap = cv2.VideoCapture(0)

def run():
    while True:
        global speaking
        response = ""
        speaking = False
        os.system('clear')
        with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt', 'r') as recognition_text:
            command = recognition_text.read()
        _, frame = cap.read()
        eyes = eye_mode(frame)
        print(eyes)
        if ('crystal' in command.lower() or eyes == "center") and command != "":
            command = command.lower().replace('crystal ', '').replace(' crystal', '').replace('crystal', '')
            with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/prompts.txt', 'r') as prompt:
                prompt_list = str(prompt.read()).split('||[end]||\n')
            with open('/Users/vatsal/Desktop/CrystalOS/Crystal3.0/completions.txt', 'r') as completions:
                completions_list = str(completions.read()).split('||[end]||\n')
            if command.lower() in prompt_list:
                print(len(completions_list), len(prompt_list))
                response = completions_list[prompt_list.index(command.lower())]
                print(response)
            if response == "" or response == None:
                try:
                    response = web_scrapper(command)
                    print(response)
                except:
                    pass

            if response == "" or response == None:
                confirmation = input('Enter Yes or no: ')
                # confirmation = 'yes'
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
                        temperature=0.4,
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

                audio = MP3("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/audio.mp3")
                audio_length = audio.info.length
                time.sleep(int(audio_length)+4)

                with open("/Users/vatsal/Desktop/CrystalOS/Crystal3.0/recognition.txt", 'w') as clearing:
                    clearing.write('')
            except Exception as e:
                print("There was an error in playing the audio: ", e)
        
        else:
            pass
                
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
def listen_continuously():
    while True:
        listen()

t1 = threading.Thread(target=listen_continuously)
t2 = threading.Thread(target=run)

t1.start()
t2.start()
# Process(target=listen_continuously).start()
# Process(target=run).start()
