import json
import uuid

from google.cloud import storage


def write_file(bucket_name, path, text):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    filename = f"temp/{str(uuid.uuid4())}.txt"
    with open(filename, 'w') as file:
        file.write(text)
    blob.upload_from_filename(filename)


def write_story(story_name, bucket, text):
    filename = str(uuid.uuid4()) + ".txt"
    path = f"kindle_books/stories/{story_name}/drafts/{filename}"
    write_file(bucket, path, text)



def get_text_file(bucket_name, filepath):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(filepath)
    return blob.download_as_string()


def get_json(bucket_name, filepath):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(filepath)
    return json.loads(blob.download_as_string())


def get_file_list(bucket, prefix):
    client = storage.Client()
    return [x.name for x in client.list_blobs(bucket, prefix=prefix)]


store.get_json('ks-stories',

               get_json(bucket, 'kindle_books/data/Behemoth Rising.json'))
