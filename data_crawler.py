
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from time import sleep
from selenium.webdriver.firefox.options import Options

picture_number = 20
options = Options()
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
 
 
def weather(city):
    res = requests.get(
        f'https://www.earthcam.com/world/{city}', headers=headers)
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
        driver.get_screenshot_as_file(filename +"_"+str(num) +".png")

    driver.quit()

if __name__ == "__main__":
    
    cities = ["ireland/dublin/"]

    for city in cities:
        path = "https://www.earthcam.com/world/" + city

        condition = weather(city)
        if condition == "Sunny": # change condition name to actual name
            name = city.replace("/", "_")
            name = name +"_"+condition
            print(name)
            web_screenshot(path, name)
        elif condition == "Foggy":  # change condition name to actual name
            name = city.replace("/", "_")
            name = "C://Data/dehaze/CustomData/"+name +"_"+condition
            print(name)
            web_screenshot(path, name)


