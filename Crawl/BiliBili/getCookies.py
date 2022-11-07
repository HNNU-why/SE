# -*- coding: utf-8 -*-
from selenium import webdriver
import time
import json
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests


def browser_initial():
    """"
    进行浏览器初始化
    """
    log_url = 'https://www.bilibili.com'
    headers = requests.get(log_url).request.headers
    # print(headers)
    resp = requests.get('https://api.bilibili.com/x/v2/reply/main?mode=3&next=1&oid=643770861&plat=1&type=1', headers=headers)
    print(resp.text)
    # chrome_options = Options()
    # chrome_options.add_argument('--headless')  # 设置无界面
    # chrome_options.add_argument('disable-infobars')
    # user_agent = (
    #     "User-Agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'")

    # chrome_options.add_argument('user-agent=%s' % user_agent)
    # browser = webdriver.Edge(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/edgedriver_win64/msedgedriver.exe")
    # return log_url, browser


def get_cookies(log_url, browser):
    """
    获取cookies保存至本地
    """
    browser.get(log_url)
    # time.sleep(300)  # 进行扫码
    # WebDriverWait(browser, 20).until(EC.presence_of_element_located((By.ID, "normalSubmit")))
    # browser.find_element_by_id('account').send_keys("123456")
    # browser.find_element_by_id('password').send_keys("123456")
    # web = browser.find_element_by_xpath('//*[@id="normalSubmit"]')
    # web.click()
    # time.sleep(2)
    dictCookies = browser.get_cookies()  # 获取list的cookies
    jsonCookies = json.dumps(dictCookies)  # 转换成字符串保存
    browser.quit()
    # with open('233_cookies.txt', 'w') as f:
    #     f.write(jsonCookies)
    # print('cookies保存成功！')
    return jsonCookies
    


if __name__ == "__main__":
    tur = browser_initial()
    # print(get_cookies(tur[0], tur[1]))

