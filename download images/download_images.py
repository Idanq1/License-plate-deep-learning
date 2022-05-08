from bs4 import BeautifulSoup
import requests

req = requests.get("https://www.google.com/search?q=%D7%9E%D7%9B%D7%95%D7%A0%D7%99%D7%95%D7%AA+%D7%9C%D7%95%D7%97%D7%99%D7%AA+%D7%A8%D7%99%D7%A9%D7%95%D7%99&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjutY2xiMb3AhWS8rsIHa8tDqEQ_AUoAXoECAEQAw&biw=1920&bih=929&dpr=1")

soup = BeautifulSoup(req.text, "html.parser")
print(soup.prettify())