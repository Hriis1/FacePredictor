import os
import tensorflow as tf
import pandas as pd

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

def map_age(age_range):
    """Map age ranges to midpoints."""
    start, end = map(int, age_range.split('-'))
    return (start + end) // 2

def preprocess_labels(label_file):
    """Read and preprocess labels from the values.txt file."""
    df = pd.read_csv(label_file)
    
    # Drop unnecessary columns
    if 'service_test' in df.columns:
        df = df.drop(columns=['service_test'])
    
    # Map age ranges to midpoints
    df['age'] = df['age'].apply(map_age)
    
    # Map gender to numeric values
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Map race to numeric categories
    race_mapping = {race: idx for idx, race in enumerate(df['race'].unique())}
    df['race'] = df['race'].map(race_mapping)
    
    return df, race_mapping

def create_data_generator(df, batch_size=32):
    """Create a TensorFlow data generator."""
    # Construct full image paths
    img_paths = df['file'].apply(lambda x: os.path.join("data/", x)).tolist()
    
    # Combine all labels into a tuple
    labels = (
        tf.convert_to_tensor(tf.keras.utils.to_categorical(df['race'], num_classes=df['race'].nunique())),
        tf.convert_to_tensor(df['gender'], dtype=tf.float32),
        tf.convert_to_tensor(df['age'], dtype=tf.float32)
    )
    
    # Create a TensorFlow Dataset
    def preprocess_image(img_path, label):
        """Load and preprocess a single image."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = img / 255.0  # Normalize
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
