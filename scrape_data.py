import requests
from bs4 import BeautifulSoup

# URL of the website
url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/2020-21'

# Send a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object from the response content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all tables with class "responsive-enabled"
tables = soup.find_all('table', class_='responsive-enabled')

# Iterate over the found tables and print their contents
for table in tables:
    print(table)