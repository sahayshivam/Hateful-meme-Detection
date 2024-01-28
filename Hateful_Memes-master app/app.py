import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
import predict
import processing.get_image_data as pg
import processing.processing as pp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    header = stream.read(512)  
    stream.seek(0)  # resets stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return show_index(filename)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/image')
def show_index(filename):
    full_filename = os.path.join(app.config['UPLOAD_PATH'], filename)
    pred, labels, objects, text = predict_single(full_filename)
    return render_template("image.html", user_image=full_filename, prediction=pred, labels=labels, objects=objects,
                           text=text)


@app.route('/predict_single')
def predict_single(path):
    

    dicti = pg.get_all(path=path)
    d = pp.preprocess(dicti)
    labels = d['labels']
    objects = d['objects']
    text = d['text']

    input = predict.vectorize(labels, objects, text)
    probas = predict.get_probabilities(input)
    print(probas)
    pred = predict.get_prediction(probas)

    if pred == 1:
        return 'This is a hateful meme', labels, objects, text
    else:
        return 'This is a safe meme', labels, objects, text


if __name__ == '__main__':
    app.debug = True
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
