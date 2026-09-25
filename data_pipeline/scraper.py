"""Polite scraper for Books to Scrape; collects at least 60 books across 3 categories."""
import time
from urllib.parse import urljoin
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/"
HEADERS = {"User-Agent": "EducationalProject/1.0 (learning project)"}
SESSION = requests.Session(); SESSION.headers.update(HEADERS)

def parse_rating(card):
    classes = card.select_one("p.star-rating").get("class", [])
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    return next((mapping[c] for c in classes if c in mapping), None)

def discover_categories(limit=3):
    response=SESSION.get(BASE, timeout=30); response.raise_for_status()
    soup=BeautifulSoup(response.text,"html.parser")
    found=[]
    for a in soup.select("div.side_categories ul li ul a"):
        name=a.get_text(strip=True)
        url=urljoin(BASE,a.get("href",""))
        if name and url not in [x[1] for x in found]: found.append((name,url))
    if len(found)<limit: raise RuntimeError("Could not discover enough categories from the catalog")
    # Choose categories with many books; category pages are paginated, so cap each at 25.
    return found[:limit]

def scrape_category(category_url, category_name, limit=None):
    rows, page_url, count = [], category_url, 0
    while page_url and (limit is None or count < limit):
        response=SESSION.get(page_url,timeout=30); response.raise_for_status()
        soup=BeautifulSoup(response.text,"html.parser")
        for card in soup.select("article.product_pod"):
            anchor=card.select_one("h3 a"); detail_url=urljoin(page_url,anchor["href"])
            detail=SESSION.get(detail_url,timeout=30); detail.raise_for_status()
            dsoup=BeautifulSoup(detail.text,"html.parser")
            price=dsoup.select_one(".price_color"); availability=dsoup.select_one(".availability")
            rows.append({"title":anchor.get("title",anchor.get_text(strip=True)),"price":price.get_text(strip=True) if price else "",
                "rating_text":next((c for c in card.select_one("p.star-rating").get("class",[]) if c!="star-rating"),""),
                "rating":parse_rating(card),"availability":availability.get_text(" ",strip=True) if availability else "",
                "category":category_name})
            count+=1
            if limit is not None and count>=limit: break
            time.sleep(.15)
        nxt=soup.select_one("li.next a")
        page_url=urljoin(page_url,nxt["href"]) if nxt else None
    return rows

def main():
    categories=discover_categories(3)
    rows=[]
    # 25/category gives >=60 if each chosen category has at least 20 items.
    for name,url in categories:
        rows.extend(scrape_category(url,name,limit=25))
    df=pd.DataFrame(rows).drop_duplicates(subset=["title","category"])
    df.to_csv("books_raw.csv",index=False)
    print(f"Saved {len(df)} unique books across {df.category.nunique()} categories to books_raw.csv")
    if len(df)<60 or df.category.nunique()<3:
        raise RuntimeError("Target not met: need >=60 books across >=3 categories. Increase per-category limit.")
if __name__=="__main__": main()
