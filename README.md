# **Recommender-System**
This system employs a sophisticated CNN-RNN-LSTM architecture to extract rich features from images and generate hashtags, leveraging historical data of image features, captions, and trending hashtags. Complementing this, the integrated Music Recommendation system tailors suggestions based on users' preferred genres. Accessible through an intuitive Streamlit interface, this platform ensures seamless interaction, making request processing and response generation effortless.
This system contributes to deliver a clever solution that will expand the reach of digital material.

## **Prerequisites**
*Anaconda Distribution (https://www.anaconda.com/distribution/)

*Python 3.7.3 (https://www.python.org/downloads/)

## **Hashtags Recommendation System**
The system design is based on attention mechanism to focus on important features of an image. The model takes an image as input and produces a one-hot encoded list of hashtags.
Image features are extracted from lower CNN layers (ENCODER). The decoder uses a LSTM that is responsible for producing a hashtag (one word) at each time step t, which is conditioned on a context vector zt, the previous hidden state ht and the previously generated hashtag. Soft attention mechanism is used to generate hashtags.

The dataset was created using the well-known Flickr8K dataset. The process involved randomly selecting 1K images and creating two csv files, one containing the image names and their associated captions, while the other file with the image names along with the trending hashtags associated with them for ground truth. Standardizing the picture inputs is necessary to improve the hashtag prediction model's effectiveness and performance. This promotes inference and guarantees the best possible learning processes. Before using CNN, this is accomplished by scaling photos to a standard size to guarantee constant input dimensions throughout the da-taset, which guarantees successful feature extraction and successful model training.

To use the features and resources of the deep learning framework, a PyTorch environment was initially established in PyCharm using Python 3.8.0. Following environment setup, a Python code seg-ment was used to segregate the preprocessed data into train, validation, and test sets. Next, the CNN-RNN-LSTM architecture-based encoder-decoder model was defined where the model utilizes CNN to extract features from images by swapping out the top fully connected layer of a pretrained ResNet-152. RNN then uses the CNN-generated feature vector to create captions, which are a series of words that characterize the image. In the same directory as the training dataset, a pickle file with a basic vocabulary wrap-per was made. The decoder's LSTM is in charge of creating hashtags. Notably, the LSTM has two layers and parameters like the word embedding vector's di-mension and the LSTM hidden states' dimension are set to 256 and 512, respec-tively. Using 80 percent of the data, which included about 800 photos, the model was trained for 250 epochs with a batch size of 128 and a learning rate of 0.01. Using an i3-8130U CPU, the training procedure took about two and a half hours to finish.

Here, the predict.py file uses the saved model (encoder and decoder) on the web interface for the uploaded image.
> [!NOTE]
> The encoder model file could not be uploaded to git due to its huge size.

## **Music Recommender System**






