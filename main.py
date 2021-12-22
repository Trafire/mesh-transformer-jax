import uuid

from gen import infer
import store
import os

def get_prompt(bucket_name, story_name):
    filepath =f"kindle books/stories/{story_name}/confirmed.txt"
    return store.get_text_file(bucket_name, filepath)


bucket_name = 'ks-stories'
json_folder_path = 'kindle_books/data'
temp = 0.9
top_p = 0.9

while True:
    json_list = store.get_file_list(bucket_name, json_folder_path)
    for json_path in json_list:
        json_data = store.get_json(bucket_name, json_path)
        story_name = json_data['story_name']
        version = json_data['version']
        prompt = get_prompt(bucket_name, story_name)
        for i in range(10):
            data = infer(top_p=top_p, temp=temp, gen_len=512, context=prompt)

            prompt += data[0]
            print(prompt)
            directory = f"kindle books/stories/{story_name}/drafts/version {version}/"
            filename = str(uuid.uuid4()) + ".txt"
            filepath = directory + filename
            print(f"saving to {filepath}")
            store.write_file(bucket_name, filepath, prompt)






