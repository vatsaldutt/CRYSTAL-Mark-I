# import sys
# import spotipy
# import spotipy.util as util

# token = spotipy.prompt_for_user_token(username='levxio53lptw44tkn4mszgra2', client_id='9283b49c483b470aad6f9328269f00d6', client_secret='daafd8792b234f2aa50824dea0f0bfb6', redirect_uri='https://accounts.spotify.com/authorize?client_id=9283b49c483b470aad6f9328269f00d6&response_type=code&redirect_uri=https%3A%2F%2Fwww.crystal-ai.com', )


# if token:
#     sp = spotipy.Spotify(auth=token)
#     results = sp.search(q="Money", type="track", limit=10)
# else:
#     print("Can't get token for")
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from pynput.keyboard import Key,Controller
from selenium import webdriver
import time

keyboard = Controller()
 
driver = webdriver.Chrome(executable_path='/Users/vatsal/Desktop/CrystalOS/Crystal/chromedrivers/chromedriver')
youtube_query = open('youtube query.txt', 'r')
query = youtube_query.read()
youtube_query.close()
driver.get("https://www.youtube.com/results?search_query="+query.replace(' ', '+'))
div = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.CLASS_NAME, "ytd-item-section-renderer"))
)
lnks=driver.find_elements_by_tag_name("a")
for lnk in lnks:
    link = lnk.get_attribute('href')
    if 'watch?v' in str(link):
        driver.get(link)
        time.sleep(1)
        keyboard.press('k')
        keyboard.release('k') 
        break