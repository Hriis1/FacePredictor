from sklearn.model_selection import train_test_split
from src.preprocess import preprocess_labels, create_data_generator
from src.model import build_model

def train_model():
    """Train the model using a data generator."""
    # Preprocess the labels
    df, race_mapping = preprocess_labels("data/values.txt")
    print("Preprocessed labels:")
    print(df.head())

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create data generators
    train_gen = create_data_generator(train_df, batch_size=32)
    val_gen = create_data_generator(val_df, batch_size=32)

    # Build the model
    model = build_model(num_race_classes=len(race_mapping))

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20
    )

    # Save the model
    model.save("multi_output_model.h5")
    print("Model saved to multi_output_model.h5")

    return model, history
