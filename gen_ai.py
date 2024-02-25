import tensorflow as tf
import nltk
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
import io

# Open the image file
img = Image.open('paris.jpg')

buffer = io.BytesIO()

# Save the scaled image to the buffer
img.save(buffer, format='JPEG')

# Getting the bytes from the buffer
buffer.seek(0)
image_bytes = buffer.read()
# Load pre-trained image features extractor
image_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Load pre-trained caption generation model
caption_model = tf.keras.models.load_model('cap-model.h5')

# Preprocess input image and generate image features
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img

# Generate caption for input image
def generate_caption(image_features):
    initial_state = caption_model.reset_states()
    caption = []
    word = '<start>'
    while word != '<end>':
        caption.append(word)
        input_sequence = tf.convert_to_tensor([[word]])
        predictions, initial_state = caption_model([image_features, input_sequence, initial_state])
        word_index = tf.argmax(predictions[0], axis=1).numpy()[0]
        word = tf.tokenizer.index_word[word_index]
    return ' '.join(caption[1:-1])  # Remove <start> and <end> tokens

# Predict Instagram metrics based on caption and image features (simplified)
def predict_instagram_metrics(caption, image_features):
    likes = len(caption) * 10  # Simplified metric
    shares = len(caption) // 5
    saves = len(caption) // 3  
    return likes, shares, saves

image_path = image_bytes
image_features = preprocess_image(image_path)
caption = generate_caption(image_features)
likes, shares, saves = predict_instagram_metrics(caption, image_features)
print("Generated Caption:", caption)
print("Expected Likes:", likes)
print("Expected Shares:", shares)
print("Expected Saves:", saves)