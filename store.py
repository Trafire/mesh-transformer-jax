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


def get_json(bucket, filepath):
    with open(f"gs://{bucket}/{filepath}", "r") as f:
        return json.load(f)

def get_file_list(bucket, prefix):
    client = storage.Client()
    return [str(x) for x in client.list_blobs(bucket, prefix=prefix)]
