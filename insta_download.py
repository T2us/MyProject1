import urllib.request
import os 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException 
from time import sleep
import insta_url
import json

def get_img_url():
    with open('./MyJson.json', 'rt', encoding="utf-8-sig") as f:
        config = json.load(f)
    temp = {'img' : [], 'video' : []}
    driver = webdriver.Chrome(config['cd_path'])
    myurl = insta_url.my_url_list(driver)
    for i in myurl:
        url = 'https://www.instagram.com/'+str(i)
        driver.get(url)

        while(1):
            sleep(1)
            pageString = driver.page_source 
            soup = BeautifulSoup(pageString, "lxml")
            try:
                videos = soup.select('.tWeCl')
                videos = soup.select('src')
                print(video)
                temp['video'] += [videos]
            except:
                imgs = soup.select('img')[1]
                imgs = imgs.attrs['src']
                if imgs:
                    temp['img'] += [imgs]
                else:
                    imgs = imgs.attrs['srcset']
                    temp['img'] += [imgs]
            try :
                driver.find_element_by_class_name("coreSpriteRightChevron").click()
            except NoSuchElementException :
                break

    driver.quit()
    #print('이미지 주소 수집완료')
    return temp
