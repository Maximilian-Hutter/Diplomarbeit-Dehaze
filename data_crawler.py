
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from time import sleep
from selenium.webdriver.firefox.options import Options
import os
from datetime import date
import time

def get_time_formated():
    currtime = time.time()
    seconds = currtime % (24 * 3600)
    hour = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60

    currtime = str(hour) + str(minutes) + str(int(seconds))
    currtime = currtime.replace(".", "-")

    today = date.today()
    today = str(today) + "_" + str(currtime)
    return today

picture_number = 20
options = Options()
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
 
 
def weather(path):
    res = requests.get(
        path, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    condition = soup.find_all(id="conditionVals")[0].getText()
    visibility = soup.find_all(id="visibleVals")[0].getText()
    visibility = visibility.split(" ")[1]

    return condition
 

def web_screenshot(path, filename):
    profile = webdriver.FirefoxProfile()
    options.page_load_strategy = 'normal'
    driver = webdriver.Firefox(options=options)
    driver.get("https://www.google.com/")
    driver.install_addon('/Users/Redmi/Downloads/adblock_for_firefox-5.3.2.xpi')
    driver.fullscreen_window()
    driver.get(path)
    windows = driver.window_handles
    driver.switch_to.window(windows[0])

    for num in range(picture_number):
        sleep(20)
        driver.get_screenshot_as_file(filename +str(num) +".png")

    driver.quit()

if __name__ == "__main__":
    
    cities = ["ireland/dublin/",
              #"usa/missouri/stlouis/",
              "virginislands/stthomas/",
              "georgia/tbilisi/",
              "france/corsica/",
              "costarica/lafortuna/",
              "canada/ontario/whitney/",
              "canada/alberta/edmonton/",
              "canada/toronto/cntower/",
              "hungary/budapest/",
              #"israel/haifa/",
              #"israel/jerusalem/",
              #"russia/moscow/",
              "spain/mallorca/",
              "usa/alaska/sitka/",
              "usa/arizona/sedona/sevenarches/",
              "usa/arizona/goodyear/",
              "usa/california/ojai/",
              "usa/california/santamonica/",
              "usa/colorado/breckenridge/",
              "usa/colorado/denver/",
              "usa/connecticut/newlondon/",
              "usa/dc/washingtonmonument/",
              "usa/dc/cherryblossoms/",
              "usa/dc/capitol/",
              "usa/georgia/atlanta/",
              "usa/illinois/chicago/",
              "usa/indiana/notredame/",
              "usa/iowa/spencer/",
              "usa/kentucky/scottsville/",
              "usa/maine/peaksisland/",
              "usa/maine/barharbor/",
              "usa/maryland/baltimore/",
              "usa/massachusetts/boston/",
              "usa/michigan/grandhaven/lakemichigan/",
              "usa/minnesota/internationalfalls/",
              "usa/minnesota/saintpaul/",
              "usa/mississippi/vicksburg/",
              "usa/missouri/kansascity/",
              "usa/montana/bigsky/",
              "usa/newhampshire/brettonwoods/",
              "usa/newyork/skyline/",
              #"usa/newyork/weehawken/",
              "usa/newmexico/ruidoso/",
              "usa/newyork/brooklynbridge/",
              #"usa/newyork/statueofliberty/",
              "usa/newyork/empirestatebuilding/",
              "usa/newyork/highline/",
              "usa/newyork/midtown/skyline/",
              "usa/newyork/rockefellercenter/",
              "usa/newyork/lakegeorge/",
              "usa/newyork/pulaski/",
              "usa/newyork/stonybrook/",
              "usa/newyork/montauk/",
              "usa/northcarolina/hickory/",
              "usa/northcarolina/manteo/",
              "usa/northdakota/lakota/",
              "usa/ohio/cincinnati/",
              "usa/ohio/cleveland/",
              "usa/oklahoma/oklahomacity/",
              "usa/oklahoma/edmond/",
              "usa/pennsylvania/pittsburgh/",
              "usa/pennsylvania/philadelphia/",
              "usa/pennsylvania/franklin/",
              "usa/southcarolina/kiawahisland/",
              "usa/tennessee/walland/blackberrymountain/",
              "usa/tennessee/nashville/",
              "usa/texas/dallas/",
              "usa/texas/seagraves/",
              "usa/texas/lakeway/",
              "usa/texas/sanantonio/",
              "usa/texas/laporte/",
              "usa/texas/houston/",
              "usa/utah/duckcreekvillage/",
              "usa/washington/seattle/",
              "usa/wyoming/jackson/"
              ]

    while True:
        fogcity = []
        for city in cities:
            usa = city.split("/")[0]
            if not usa == "usa":
                path = "https://www.earthcam.com/world/" + city
            else: path = "https://www.earthcam.com/" + city

            condition = None
            try:
                condition = weather(path)
            except:
                print(path + " not available")
                print("\n")


            if condition == "A Few Clouds":  # names are Mist (small) haze(medium) Fog(max) A Few Clouds (gt)
                fogcity.append(city)
                name = city.replace("/", "_")
                name = "C://Data/dehaze/CustomData/"+name+"/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name +condition + "/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name + str(get_time_formated())
                web_screenshot(path, name)

            if condition == "Fog":  # names are Mist (small) haze(medium) Fog(max)
                fogcity.append(city)
                name = city.replace("/", "_")
                name = "C://Data/dehaze/CustomData/"+name+"/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name +condition + "/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name + str(get_time_formated())
                web_screenshot(path, name)

            if condition == "Mist":  # names are Mist (small) haze(medium) Fog(max)
                fogcity.append(city)
                name = city.replace("/", "_")
                name = "C://Data/dehaze/CustomData/"+name+"/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name +condition + "/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name + str(get_time_formated())
                web_screenshot(path, name)

            if condition == "Haze":  # names are Mist (small) haze(medium) Fog(max)
                fogcity.append(city)
                name = city.replace("/", "_")
                name = "C://Data/dehaze/CustomData/"+name+"/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name +condition + "/"
                if not os.path.isdir(name):
                    os.mkdir(name)
                name = name + str(get_time_formated())
                web_screenshot(path, name)


