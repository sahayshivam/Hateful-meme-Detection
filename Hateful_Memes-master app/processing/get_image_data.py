import io
from google.cloud import vision
import os

password_file = "/Users/sohamjain/Downloads/Hateful_Memes-master app/processing/My Project-e17cafef88b1.json"  # Write here yours
# path = 'uploads/01235.png'


def get_all(path, password_file=password_file):
    # Set up google vision
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = password_file
    client = vision.ImageAnnotatorClient()

    # Read image
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ret_text = [text.description for text in texts]
    print(ret_text)

    # Label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations
    ret_label = [label.description for label in labels]
    print(ret_label)

    # Object detection
    objects = client.object_localization(
        image=image).localized_object_annotations
    ret_object = [(object_.name, object_.score) for object_ in objects]
    print(ret_object)

    return {'path': path, 'text': ret_text, 'labels': ret_label, 'objects': ret_object}
