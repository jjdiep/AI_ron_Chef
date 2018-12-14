# recipe_crawler.py
# version_2.0
from bs4 import BeautifulSoup
from collections import defaultdict
import requests

# input ingredient
ingredient = ['cheese', 'egg', 'garlic','lettuce','potato']
# initialize query prefix
query = "https://www.allrecipes.com/search/results/?wt="

# fill the query by each ingredient
for item in ingredient:
	if ingredient[-1] != item:
		query += str(item) + '%20'
	else:
		query += str(item) + '&sort=re'

# get the page of search result
search_page = requests.get(query, timeout=5)
# fetching content of the search result
search_content = BeautifulSoup(search_page.content, "html.parser")
# initialize top-5 recipe website
top_5 = []
# search all links
for link in search_content.find_all('a'):
	# when list is full break
	if len(top_5) == 5:
		break
	# get recipe links
	current = link.get('href')
	# make sure it is not none
	if current is not None:
		# initialize link prefix
		prefix = "https://www.allrecipes.com/recipe/"
		# add recipe links
		if prefix in current and current not in top_5:
			top_5.append(current)

# initialize recipe title, ingredient list, and direction list
recipe_name = []
recipe_ingredient = defaultdict(list)
recipe_direction = defaultdict(list)
recipe_nutrition = defaultdict(str)
# get top-5 recipe links
for recipe in top_5:
	# get the page of recipe
	page = requests.get(recipe, timeout=5)
	# fetching content of the recipe
	page_content = BeautifulSoup(page.content, "html.parser")
	# find title
	title = page_content.find('h1', {'class':'recipe-summary__h1'})
	# add title to tne recipe name list
	recipe_name.append(title.text)
	# find all ingredients
	ingred = page_content.find_all('span', {'class':'recipe-ingred_txt added'})
	# add all ingredients to the list
	for i in range(len(ingred)):
		recipe_ingredient[title.text].append(ingred[i].text)
	# find full directions
	direction = page_content.find_all('span', {'class':'recipe-directions__list--item'})
	# add full directions to the list
	for i in range(len(direction)):
		recipe_direction[title.text].append(direction[i].text)
	# add nutrition info
	nutrition = page_content.find('div', {'class':'nutrition-summary-facts'})
	recipe_nutrition[title.text] = nutrition.text.replace('\n', '').replace(';', '; ').replace('Full nutrition', '').replace('  ', ' ')
	

# for each recipe
for i in range(len(recipe_name)):
	# open a file called recipe_x
	f = open("recipe/recipe_"+str(i+1), 'w')
	# write recipe name
	f.write(recipe_name[i]+'\n')
	f.write('\n')
	# write all ingredients
	f.write('Ingredients:' +'\n')
	for j in range(len(recipe_ingredient[recipe_name[i]])):
		f.write(str(recipe_ingredient[recipe_name[i]][j])+'\n')
	f.write('\n')
	# write full directions
	f.write('Directions:' +'\n')
	for k in range(len(recipe_direction[recipe_name[i]])):
		if (k+1) == len(recipe_direction[recipe_name[i]]):
			break
		f.write(str(k+1)+'. '+str(recipe_direction[recipe_name[i]][k])+'\n')
	# write nutrition info
	f.write('Nutrition Facts:' +'\n')
	f.write(str(recipe_nutrition[recipe_name[i]])+'\n')
	f.write('\n')
	# write source
	f.write('Source:'+'\n')
	f.write(str(top_5[i]))
	f.close()

