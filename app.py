import argparse

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename, redirect
import os
from predict import main as generate_hashtags
from musicTest import maing as recommend
from musicTest import mainf as recd_song


app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Call predict.py to generate hashtags
        args = argparse.Namespace(
            vocab_path='path/to/your/vocab.pkl',
            encoder_path='path/to/your/encoder/model.pth',
            decoder_path='path/to/your/decoder/model.pth',
            embed_size=256,
            hidden_size=512,
            num_layers=2
        )
        hashtags = generate_hashtags(args, filepath)

        # Get user preferences
        genre = request.form.get('genre')
        song = request.form.get('song')
        # Call musicTest.py functions based on user preferences
        if genre:
            genre_recommendations = recommend(genre)
        else:
            genre_recommendations = []
        if song:
            song_recommendations = recd_song(song)
        else:
            song_recommendations = []
        return render_template('result.html', filename=filename, hashtags=hashtags, genre_recommendations=genre_recommendations, song_recommendations=song_recommendations)
    else:
        return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)



# from __future__ import division, print_function
# import sys
# import os
# import glob
# import re
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import argparse
# import pickle
# from torchvision import transforms
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
# from PIL import Image
# import ast
# import json
# from collections import Counter
# from torchvision.io import read_image, ImageReadMode
# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
#
# # Define a flask app
# app = Flask(__name__)
#
# # Device configuration
# device = torch.device('cpu')
#
#
# def load_image(image_path, transform=None):
#     # image = Image.open(image_path).convert('RGB')
#     image = read_image(image_path, mode=ImageReadMode.RGB)
#     image = image.resize([224, 224], Image.LANCZOS)
#
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
#
#     return image
#
#
# def process(args):
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as func:
#         vocab = pickle.load(func)
#
#     # calling saved models
#     encoder = EncoderCNN(args.embed_size).eval()
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)
#
#     # Load the trained model parameters
#     encoder.load_state_dict(torch.load(args.encoder_path, map_location=lambda storage, loc: storage))
#     decoder.load_state_dict(torch.load(args.decoder_path, map_location=lambda storage, loc: storage))
#
#     # Prepare an image
#     image = load_image(args.image, transform)
#     image_tensor = image.to(device)
#
#     # Generate the hashtag from the image
#     feature = encoder(image_tensor)
#     sampled_ids = decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()
#
#     # Convert word_ids to words
#     sampled_hashtags = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         sampled_hashtags.append(word)
#         if word == '<end>':
#             break
#     sentence = ' '.join(sampled_hashtags)
#
#     return sentence
#
#
# def predict(img_path):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, default=img_path, help='input image for generating caption')
#     parser.add_argument('--encoder_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\encoder-40-1.ckpt')
#     parser.add_argument('--decoder_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-40-1.ckpt',
#                         help='path for trained decoder')
#     parser.add_argument('--vocab_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl',
#                         help='path for vocabulary wrapper')
#
#     # Model parameters (should be same as paramters in train.py)
#     parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
#     args = parser.parse_args()
#     predicted_hashtags = process(args)
#
#     return predicted_hashtags
#
#
# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']
#
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, secure_filename(f.filename))
#         f.save(file_path)
#
#         # Make prediction
#         preds = predict(file_path)
#         hashtags_list = preds.split(' ')
#         hashtags_list.remove('<start>')
#         hashtags_list.remove('<end>')
#         tags = []
#         for tag in hashtags_list:
#             if tag not in tags:
#                 tags.append(tag)
#         text = ""
#         for tag in tags:
#             text = text + '  #' + str(tag)
#         text = ' ' + text
#         return text
#     return None
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
