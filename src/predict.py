import tensorflow as tf

def load_model():
    """Load the trained model."""
    return tf.keras.models.load_model('multi_output_model.h5')

def predict_image(model, image_path, race_mapping):
    """Make predictions for a single image."""
    # Preprocess the image
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_race = tf.argmax(predictions[0], axis=1).numpy()[0]
    predicted_gender = round(predictions[1][0][0])
    predicted_age = int(predictions[2][0][0])

    print(f"Predicted Race: {race_mapping[predicted_race]}")
    print(f"Predicted Gender: {'Male' if predicted_gender == 1 else 'Female'}")
    print(f"Predicted Age: {predicted_age}")
