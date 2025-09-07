import torch
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import ast
import json
from collections import Counter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    if not os.path.exists(image_path):
        # If the file doesn't exist, try appending common image extensions
        extensions = ['.jpg', '.jpeg', '.png']
        for ext in extensions:
            temp_path = image_path + ext
            if os.path.exists(temp_path):
                image_path = temp_path
                break
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence


if __name__ == '__main__':
    test_img_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\images'
    test_captions_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\tag_list.txt'
    test_data = {}
    with open(test_captions_path, "r") as data:
        next(data)  # Skip the header line
        for line in data:
            parts = line.strip().split(',')
            img_name = parts[0]
            tags = parts[1].split()
            test_data[img_name] = tags
    print(data)
    # Create a file to save the results
    save_file_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\result.txt'
    save_in_file = open(save_file_path, "w")

    # Initialize counter
    counter = Counter()

    # Iterate over test data
    for id, info in test_data.items():
        image_name = info[1]
        image_extension = os.path.splitext(image_name)[1]  # Extract file extension
        complete_path = os.path.join(test_img_path, image_name)
        if not os.path.exists(complete_path):
            # If the file with the given name doesn't exist, try adding the extension
            complete_path = os.path.join(test_img_path, image_name + image_extension)
        caption = info[2]

        # Parse command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str, default=complete_path, help='input image for generating caption')
        parser.add_argument('--encoder_path', type=str,
                            default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\encoder-40-1.ckpt',
                            help='path for trained encoder')
        parser.add_argument('--decoder_path', type=str,
                            default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-40-1.ckpt',
                            help='path for trained decoder')
        parser.add_argument('--vocab_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl',
                            help='path for vocabulary wrapper')

        # Model parameters (should be same as parameters in train.py)
        parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
        parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
        parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
        args = parser.parse_args()

        # Generate caption for the image
        predicted_caption = main(args)

        # Write results to file
        save_in_file.write(complete_path)
        save_in_file.write(',')
        save_in_file.write('original caption : ' + str(caption))
        save_in_file.write(',')
        save_in_file.write('predicted caption : ' + str(predicted_caption))
        save_in_file.write('\n')

    save_in_file.close()




# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import pickle
# import os
# from torchvision import transforms
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
# from PIL import Image
# import ast
# import json
# from collections import Counter
#
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def load_image(image_path, transform=None):
#     image = Image.open(image_path)
#     image = image.resize([224, 224], Image.LANCZOS)
#
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
#
#     return image
#
#
# def main(args):
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#
#     # Build models
#     encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)
#
#     # Load the trained model parameters
#     encoder.load_state_dict(torch.load(args.encoder_path))
#     decoder.load_state_dict(torch.load(args.decoder_path))
#
#     # Prepare an image
#     image = load_image(args.image, transform)
#     image_tensor = image.to(device)
#
#     # Generate a caption from the image
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
#     # Print out the image and the generated caption
#     print(sentence)
#     image = Image.open(args.image)
#
#     return sentence
#     # plt.imshow(np.asarray(image))
#
#
# if __name__ == '__main__':
#     test_img_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\images\momoyandesign.png'
#     test_captions_path =r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\tag_list.txt'
#     with open(test_captions_path, "r") as data:
#         test_data = ast.literal_eval(data.read())
#     counter = Counter()
#     all_test_img_paths = []
#     all_test_captions = []
#     ids = test_data.keys()
#     save_in_file = open(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\result.txt', "w")
#     for i, id in enumerate(ids):
#         image_name = test_data[id][1]
#         print("\nimage name : {}".format(test_data[id][1]))
#         complete_path = test_img_path + test_data[id][0] + '_' + test_data[id][1]
#         print("\nimage path : {}".format(complete_path))
#         # caption='<start> '+test_data[id][2]+' <end>'
#         caption = test_data[id][2]
#
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--image', type=str, default=complete_path, help='input image for generating caption')
#         parser.add_argument('--encoder_path', type=str,
#                             default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\encoder-40-330.ckpt',
#                             help='path for trained encoder')
#         parser.add_argument('--decoder_path', type=str,
#                             default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-40-330.ckpt',
#                             help='path for trained decoder')
#         parser.add_argument('--vocab_path', type=str, default=r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl',
#                             help='path for vocabulary wrapper')
#
#         # Model parameters (should be same as paramters in train.py)
#         parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
#         parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
#         parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
#         args = parser.parse_args()
#         predicted_caption = main(args)
#         save_in_file.write(complete_path)
#         save_in_file.write(',')
#         text = 'original caption : ' + str(caption)
#         save_in_file.write(text)
#         save_in_file.write(',')
#         text = 'predicted caption : ' + str(predicted_caption)
#         save_in_file.write(text)
#         save_in_file.write('\n')
#
#     save_in_file.close()