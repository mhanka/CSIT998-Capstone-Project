import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np



auctionresultslinks=[]
auctiondatalinks=[]
auctionimglinks=[]
artistnamedata=[]
artdetails=[]
soldprice=[]
estimateprice=[]



for i in range (10,13):
  url="https://www.christies.com/en/results?year=2020&month="+str(i)
  options = Options()
  options.headless = True
  options.add_argument("--window-size=1920,1080")
  options.add_argument("--start-maximized")
  options.add_argument("--disable-gpu")
  options.add_argument("--ignore-certificate-errors")
  options.add_argument("--disable-extensions")
  options.add_argument("--disable-infobars")
  options.add_argument("--no-sandbox")
  options.add_argument("--disable-dev-shm-usage")

  Driver_path= 'C:\\Users\\sailajamon\\Documents\\schoteby_2020_contemporaryArt_stats\\chromedriver.exe'
  driver = webdriver.Chrome(options=options,executable_path=Driver_path)

  try: 
    driver.get(url)
  except TimeoutException:
    pass
  except WebDriverException:
    pass
  time.sleep(2)  
  driver.find_element(By.XPATH,"//*[@id='onetrust-accept-btn-handler']").click()
  driver.implicitly_wait(30)
  driver.find_element(By.XPATH,"//*[@id='close_signup']").click()
  driver.maximize_window()
  driver.find_element(By.XPATH,"//body/main[1]/div[3]/section[1]/div[1]/chr-calendar[1]/div[1]/section[1]/chr-sticky-wrapper[1]/chr-filter-block-events[1]/chr-filter-block[1]/chr-accordion[1]/div[1]/chr-accordion-item[4]/div[1]/div[1]/div[1]/span[1]/div[1]").click()
  checkboxelements=driver.find_elements(By.CLASS_NAME,"chr-checkbox__input")
  for element in checkboxelements:
    valuedetails=element.get_attribute('value')
    time.sleep(3)
    if(valuedetails=="category_7"):
      driver.execute_script("arguments[0].click();", element)
    elif(valuedetails=="category_11"):
      driver.execute_script("arguments[0].click();", element)
      break   
    else:
      continue
  soup=BeautifulSoup(driver.page_source,'html.parser')
  auctionlink=soup.find_all("a",attrs={'class':'chr-event-tile__title'})
  for link in auctionlink:
    links=link.get("href")
    auctionresultslinks.append(links)

print(auctionresultslinks)
for linker in auctionresultslinks:
    url=linker
    driver = webdriver.Chrome(options=options,executable_path=Driver_path)
    try:   
      driver.get(url)  
    except TimeoutException:
      pass
    except WebDriverException:
      pass
    time.sleep(1)
    scroll_pause_time=2
    screen_height = driver.execute_script("return window.screen.height;")   
    i = 1
    while True:
      driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))
      i += 1
      time.sleep(scroll_pause_time)
      scroll_height = driver.execute_script("return document.body.scrollHeight;")  
      if (screen_height) * i > scroll_height:
        break 
    soup=BeautifulSoup(driver.page_source,'html.parser')
    for parentdet in soup.find_all("div",attrs={'class':'col-12 col-md-4 col-lg-3 col-xl-2 chr-lot-tile-container'}):
      a_tag=parentdet.find("a",attrs={'class':'chr-lot-tile__link'})  
      artlink=a_tag.attrs['href']
      auctiondatalinks.append(artlink)
      time.sleep(1)

print(auctiondatalinks)      
for artdata in auctiondatalinks:
  driver = webdriver.Chrome(options=options,executable_path=Driver_path)
  url=artdata
  try:
    driver.get(url)
  except TimeoutException:
    pass
  except WebDriverException:
    pass
  time.sleep(2)
  try:
    artistname=driver.find_element(By.XPATH,"//*[@class='chr-lot-header__artist-name']").text.strip()
      
  except NoSuchElementException:
    pass
  artistnamedata.append(artistname)
  try:    
    pricereleased=driver.find_element(By.XPATH,"//chr-lot-header-dynamic-content/div[1]/div[1]/div[1]/span[2]").text.strip()
     
  except NoSuchElementException:
    pass 
  soldprice.append(pricereleased)  
  try:
    priceestimate=driver.find_element(By.XPATH,"//chr-lot-header-dynamic-content/div[1]/div[1]/div[2]/div[2]/span[1]").text.strip()
     
  except NoSuchElementException:
    pass 
  estimateprice.append(priceestimate)   
  try:
    imglink=driver.find_element(By.TAG_NAME,"img").get_attribute("src")
     
  except NoSuchElementException:
    pass
  auctionimglinks.append(imglink) 
  try:   
    auctiondetails=driver.find_element(By.XPATH,"//*[@class='content-zone chr-lot-section__accordion--content']").text.strip()
    #datadetails=auctiondetails.split("\n")
  except NoSuchElementException:
    pass
  artdetails.append(auctiondetails)
  time.sleep(1)

        
  
  data={'ArtistName':artistnamedata,'EstimatePrice':estimateprice,'soldprice':soldprice,'Artdetails':artdetails,'imglinks':auctionimglinks}
  datasets=pd.DataFrame(data=data)
      
  cols=['ArtistName','EstimatePrice','soldprice','Artdetails','imglinks']
  datasets=datasets[cols]

  filename="C:\\Users\\sailajamon\\Documents\\Christie_python_script\\2020_Christie_Dataset_2.xlsx"
  wks_name="Data"

  writer=pd.ExcelWriter(filename, engine='openpyxl')
  datasets.to_excel(writer, wks_name, index=False)
  writer.save()
