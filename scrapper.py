from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

import time
import pandas as pd
import re
import streamlit as st
from selenium.common.exceptions import StaleElementReferenceException


@st.cache_data
def builder():
    # Base URL without the page parameter
    base_url = "https://www.vivareal.com.br/aluguel/rio-grande-do-norte/natal/apartamento_residencial/?pagina={page}#onde=,Rio%20Grande%20do%20Norte,Natal,,,,,city,BR>Rio%20Grande%20do%20Norte>NULL>Natal,,,&preco-ate=3000&preco-total=sim"

   

    # Function to scrape data for a single page
    def scrape_page(page):
         # Setting up the Chrome WebDriver
        options = Options()
        #options.binary_location = "/usr/bin/google-chrome"
        #options.add_argument("--headless")  # Add headless mode if necessary
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-features=NetworkService")
        options.add_argument("--window-size=1920x1080")
        options.add_argument("--disable-features=VizDisplayCompositor")
        # Initialize the WebDriver
        service = Service(ChromeDriverManager().install())
        wd = webdriver.Chrome(service=service, options=options)
        # Open the URL for the current page
        url = base_url.format(page=page)
        wd.get(url)
        
        # Wait for the page to load
        wait = WebDriverWait(wd, 40)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-list")))

        # A retry mechanism for stale elements
        def get_elements_with_retries(by, value, retries=3):
            for attempt in range(retries):
                try:
                    elements = wd.find_elements(by, value)
                    return [element.text for element in elements]
                except StaleElementReferenceException:
                    print(f"Stale element reference exception on attempt {attempt + 1}. Retrying...")
                    time.sleep(2)  # Wait before retrying
            return []

        # Scrape the data
        prices = get_elements_with_retries(By.CLASS_NAME, "property-card__price")
        locations = get_elements_with_retries(By.CLASS_NAME, "property-card__address")
        sizes = get_elements_with_retries(By.CLASS_NAME, "js-property-card-detail-area")
        quartos = get_elements_with_retries(By.CLASS_NAME, "property-card__detail-room")
        banheiros = get_elements_with_retries(By.CLASS_NAME, "property-card__detail-bathroom")
        link_elements = wd.find_elements(By.CLASS_NAME, "property-card__content-link")


        
        def extract_price_and_condo(card):                      
            try:
                condo_element = card.find_element(By.CLASS_NAME, "js-condo-price")                             
                condo_text = condo_element.text.strip().split()[1]  # Extracting the condo fee as text
                condo_price = float(condo_text.replace('R$', '').replace('.', '').replace(',', '.'))                
            except:
                condo_price = 0  # If no condo fee is found

            return condo_price

        cards = wd.find_elements(By.CLASS_NAME, "property-card__values")
        # Extracting prices (rent + condo) for all cards
        extras = [extract_price_and_condo(card) for card in cards]        
        links = [link.get_attribute("href") for link in link_elements]

        # Return data in a DataFrame format
        return pd.DataFrame({
            "Aluguel": prices,
            "Extras": extras,
            "Localização": locations,
            "Tamanho": sizes,
            "Quartos": quartos,
            "Banheiros": banheiros,
            "Link": links
        })

        wd.quit()

    # List to store all scraped DataFrames
    df_list = []

    # Loop through the first 5 pages
    for page in range(1, 16):
        df_page = scrape_page(page)    
        df_list.append(df_page)
        

    # Combine all pages into one DataFrame
    df_apartments = pd.concat(df_list, ignore_index=True)

    # Extract the neighborhood
    def extract_neighborhood(location):
        # Try to match pattern with " - " separating street and neighborhood
        match = re.search(r' - ([^,]+),', location)
        
        # If the first pattern doesn't match, assume location starts with the neighborhood
        if not match:
            match = re.search(r'^([^,]+),', location)
        
        # Return the matched neighborhood or None if not found
        if match:
            return match.group(1)
        return None

    df_apartments['Bairro'] = df_apartments['Localização'].apply(extract_neighborhood)
    
    st.write(df_apartments)
    df_apartments.to_csv('apartments.csv', index=False)
    return df_apartments
    # Close the browser

