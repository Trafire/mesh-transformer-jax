import csv
import json
import random
def get_genres(data):
    d = data.split(':')
    index = 0
    genre_list = []
    for c in d:
        if index % 2:
            genre = c.split(',')[0].replace('}', '')

            if genre == "Roman \u00e0 clef":
                genre = "Romance"
            genre_list.append(genre.strip())
        index += 1
    if not genre_list:
        genre_list = ["Fiction"]
    return ','.join(genre_list).replace('"','')







def get_data(filename):
    data = []
    with open(filename, encoding='utf-8') as file:
        csvreader = csv.reader(file, delimiter='\t')
        for row in csvreader:
            title = row[2]
            genre = get_genres(row[5])
            summary = row[6]
            data.append({'title': title, 'genre': genre, 'summary': summary})
    return data

def create_record(title, genre, summary):
    return f"""Title: {title}
Genre: {genre}
Summary: {summary} 
<|endoftext|>"""


filename = 'data/booksummaries.txt'

data = get_data(filename)

random.shuffle(data)
index = int(len(data) * .85)
train = data[:index]
eval = data[index:]


with open('data/booksummaries_eval.txt','w', encoding='utf-8') as file:
    for d in eval:
        if 'Non-fiction' not in d['genre']:
            file.write(create_record(**d))

with open('data/booksummaries_train.txt','w', encoding='utf-8') as file:
    for d in train:
        if 'Non-fiction' not in d['genre']:
            file.write(create_record(**d))



