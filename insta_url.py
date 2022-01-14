from bs4 import BeautifulSoup
import selenium.webdriver as webdriver
from selenium import webdriver
import urllib.parse
from urllib.request import Request, urlopen
from time import sleep
import json

def my_url_list(driver):
    
    with open("C:/inetpub/flask/MyJson.json", 'rt', encoding="utf-8-sig") as f:
        config = json.load(f)

    url1 = "https://www.instagram.com"
    url = "https://www.instagram.com/{}" .format(config['keyword'])
    
    limit = config['wish_num']
#    user_id=config['user_id']
#    user_passwd=config['user_passwd']
    user_id="인스타그램 아이디 입력"
    user_passwd="인스타그램 비밀번호 입력"
    
    print("login start")

    driver.get(url1)
    sleep(3)

    driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[1]/div/label/input').send_keys(user_id)
    driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[2]/div/label/input').send_keys(user_passwd)

    sleep(5)

    driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[3]').click()

    sleep(10)

    #print("login success")

    sleep(10)
    
    driver.get(url) 
    sleep(5)


    SCROLL_PAUSE_TIME = 1.0
    reallink = []

    while True:
        pageString = driver.page_source
        bsObj = BeautifulSoup(pageString, "lxml")
        
        for link1 in bsObj.find_all(name="div",attrs={"class":"Nnq7C weEfm"}):
                title = link1.select('a')[0] 
                if title:
                    real = title.attrs['href']
                    reallink.append(real)

                if limit != 0 and limit == len(reallink):
                    stop = True
                    break
                
                title = link1.select('a')[1]
                if title: 
                    real = title.attrs['href']
                    reallink.append(real) 

                if limit != 0 and limit == len(reallink):
                    stop = True
                    break

                title = link1.select('a')[2] 
                if title:
                    real = title.attrs['href']
                    reallink.append(real)
                
                if limit != 0 and limit == len(reallink):
                    stop = True
                    break
        if stop == True:
            break

        last_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
                
            else:
                last_height = new_height
                continue

    reallink = list(set(reallink))
    reallinknum = len(reallink)
    #print("총"+str(reallinknum)+"개의 URL 수집")
    return reallink

