# import requests
# from bs4 import BeautifulSoup

# def scrape_articles(query, num_articles=5):
#     url = f"https://www.google.com/search?q={query}"
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         print(f"Request failed with status code: {response.status_code}")
#         return []

#     soup = BeautifulSoup(response.text, 'html.parser')

#     articles = []
#     print("HTML Content:", soup.prettify()[:1000])  # Print a portion of the HTML for debugging

#     # Google search result might be within 'div' with class 'g' or a similar structure
#     search_results = soup.find_all('div', class_='tF2Cxc')
#     print(f"Found {len(search_results)} search result blocks")

#     for result in search_results[:num_articles]:
#         print("Found div with class 'tF2Cxc':", result)  # Print each found div for debugging

#         # Extracting link, title, and snippet from the search result
#         link_tag = result.find('a')
#         title_tag = result.find('h3')
#         snippet_tag = result.find('div', class_='IsZvec')

#         if link_tag and title_tag and snippet_tag:
#             link = link_tag['href']
#             title = title_tag.get_text()
#             snippet = snippet_tag.get_text()

#             articles.append({
#                 'title': title,
#                 'link': link,
#                 'snippet': snippet
#             })

#     print("Articles found:", articles)  # Print the list of found articles for debugging
#     return articles

# # Example usage
# query = 'Soybean___healthy plant disease'
# articles = scrape_articles(query)
# print(articles)

import numpy as np

# Sample data
x = np.array([0, 1, 2, 3, 4])
y = np.array([10, 15, 5, 20, 10])

# Calculate slope (m)
m = (np.mean(y) - np.mean(x) * np.cov(x, y)[0, 1]) / np.var(x)

# Calculate intercept (b)
b = np.mean(y) - m * np.mean(x)

# Print slope and intercept
print("Slope (m):", m)
print("Intercept (b):", b)
