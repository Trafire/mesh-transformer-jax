import json
import uuid

from google.cloud import storage


def write_story(story_name, bucket, text):
    filename = str(uuid.uuid4()) + ".txt"
    with open(f"gs://{bucket}/kindle_books/stories/{story_name}/drafts/{filename}", "w") as f:
        f.write(text)


def get_text_file(bucket, filepath):
    with open(f"gs://{bucket}/{filepath}", "r") as f:
        return f.read()


def get_json(bucket_name, filepath):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(filepath)
    return json.loads(blob.download_as_string())

def get_file_list(bucket, prefix):
    client = storage.Client()
    return [x.name for x in client.list_blobs(bucket, prefix=prefix)]
