from __future__ import unicode_literals
import sqlite3
import cv2
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary
from music import recommend, recd_song
# from Recognize import voice_recognition

# Load the vocabulary wrapper
with open(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load the trained model
encoder = EncoderCNN(256).eval()  # Set embedding size to 256
decoder = DecoderRNN(256, 512, len(vocab), 2)  # Adjust hidden size and number of layers if needed
encoder.load_state_dict(
    torch.load(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\Model.ckpt\encoder-250-1.ckpt', map_location=torch.device('cpu')))
decoder.load_state_dict(
    torch.load(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\Model.ckpt\decoder-250-1.ckpt', map_location=torch.device('cpu')))
encoder.eval()
decoder.eval()

# Define image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Function to generate caption from image
def generate_caption(image):
    image_tensor = transform(image).unsqueeze(0)
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return ' '.join(sampled_caption)


# def process_video(video_file):
#     cap = cv2.VideoCapture(str(video_file))
#     hashtags = set()
#     frame_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert frame to PIL image
#         frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#         # Generate caption and extract hashtags
#         caption = generate_caption(frame_pil)
#         hashtags.update(caption.split())
#
#         # # Save frame as JPG file
#         # frame_count += 1
#         # frame_filename = f'frame_{frame_count}.jpg'
#         # cv2.imwrite(frame_filename, frame)
#         #
#         # # Load frame as PIL image
#         # frame_pil = Image.open(frame_filename)
#         #
#         # # Generate caption and extract hashtags
#         # caption = generate_caption(frame_pil)
#         # hashtags.update(caption.split())
#
#     cap.release()
#     return hashtags


# def voice_recognition(filename):
#     FRAME_RATE = 16000
#     CHANNELS = 1
#     vosk_model_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vosk-recasepunc-en-0.22'
#
#     model = vosk.Model(model_name='en', model_path=vosk_model_path)
#     rec = vosk.KaldiRecognizer(model, FRAME_RATE)
#     rec.SetWords(True)
#     mp4 = mp.VideoFileClip(filename)
#     mmp3 = mp4.audio
#     mp3 = AudioSegment.from_mp3(mmp3)
#     mp3 = mp3.set_channels(CHANNELS)
#     mp3 = mp3.set_frame_rate(FRAME_RATE)
#
#     step = 45000
#     transcript = ""
#     for i in range(0, len(mp3), step):
#         print(f"Progress: {i / len(mp3)}")
#         segment = mp3[i:i + step]
#         rec.AcceptWaveform(segment.raw_data)
#         result = rec.Result()
#         text = json.loads(result)["text"]
#         transcript += text
#
#     return transcript


# Streamlit app UI
st.title('TrendPulse')
st.subheader("Content's Best Guide, TrendPulse by Your Side!")

# Navigation buttons
page = st.sidebar.radio("Navigation", ["HASHtheTAGS", "VideOpulse", "MUSIC-^-PULSE"])

if page == "HASHtheTAGS":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)  # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Generate Caption'):
            caption = generate_caption(image)
            st.write('**Generated Hashtags:**', caption)

# elif page == "VideOpulse":
#     uploaded_video = st.file_uploader("Upload the video", type=["mp4"])
#     st.button('Generate Hashtags')
#     if uploaded_video is not None:
#         hashtags = process_video(uploaded_video)
#         st.write('**Generated Hashtags:**', hashtags)

# elif page == "TRANSCRIPTOR":
#     uploaded_video = st.file_uploader("Upload the video", type=["mp4"])
#     st.button('ENTER')
#     if uploaded_video is not None:
#         tpt = voice_recognition(uploaded_video)
#         st.write('**TRANSCRIPTIONS**', tpt)

elif page == "MUSIC-^-PULSE":
    st.title('MUSIC-^-PULSE')
    recommendation_type = st.radio("Select Recommendation Type", ["Genre", "Songs"])

    if recommendation_type == "Genre":
        genres = [
            "Trending Genres",
            "BollywoodDance",
            "BollywoodDanceRomantic",
            "BollywoodRomantic",
            "Bollywood",
            "BollywoodRomanticSad",
            "BollywoodDevotional",
            "BollywoodDanceSad",
            "BollywoodSad",
            "BollywoodMotivational",
            "BollywoodRomanticSadSensual",
            "BollywoodRomanticSensual",
            "BollywoodDancePatriotic",
            "BollywoodMotivationalPatriotic",
            "BollywoodDanceSensual",
            "BollywoodDevotionalSad",
            "BollywoodDanceMotivationalPatriotic",
            "BollywoodPatrioticSad"
        ]

        x = st.selectbox("Genres :", genres)
        pickle.load(open('musicrec.pkl', 'rb'))
        pickle.load(open('similarities.pkl', 'rb'))
        # favorite_genre = st.text_input("FOR RECOMMENDATIONS BASED ON GENRE, ENTER YOUR FAVORITE GENRE")
    if st.button("Get Recommendations"):
        recommendations = recommend(x)
        st.write("Recommended Songs:", recommendations)

    elif recommendation_type == "Songs":
        favorite_song = st.text_input("FOR RECOMMENDATIONS BASED ON SONGS, SELECT YOUR FAVORITE SONG")
        if st.button("Get Recommended songs"):
            recommendations = recd_song(favorite_song)
            st.write("Recommended Songs:", recommendations)

# import argparse
# import os
# import torch
# from torchvision import transforms
# from PIL import Image
# import pickle
# from model import EncoderCNN, DecoderRNN
# from build_vocab import Vocabulary
#
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def load_image(image_path, transform=None):
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize([224, 224], Image.LANCZOS)
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
#     return image
#
#
# def main(args):
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#
#     # Load the trained model
#     encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)
#     encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
#     decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))
#     encoder.eval()
#     decoder.eval()
#
#     # Prepare image
#     image = load_image(args.image, transform)
#     image_tensor = image.to(device)
#
#     # Generate caption from image
#     feature = encoder(image_tensor)
#     sampled_ids = decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
#
#     # Convert word_ids to words
#     sampled_caption = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         sampled_caption.append(word)
#         if word == '<end>':
#             break
#     sentence = ' '.join(sampled_caption)
#
#     # Print the predicted caption
#     print("Predicted caption:", sentence)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True,
#                         help='input image for generating caption')
#     parser.add_argument('--encoder_path', type=str, required=True,
#                         default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-150-1.ckpt',
#                         help='path for trained encoder')
#     parser.add_argument('--decoder_path', type=str, required=True,
#                         default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-150-1.ckpt',
#                         help='path for trained decoder')
#     parser.add_argument('--vocab_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl',
#                         help='path for vocabulary wrapper')
#     parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
#     args = parser.parse_args()
#     main(args)
#

