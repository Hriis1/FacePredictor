from src.train import train_model
from src.predict import load_model, predict_image

def main():
    choice = input("Choose an option:\n1. Train the model\n2. Predict an image\n")
    
    if choice == '1':
        print("Training the model...")
        train_model()
    elif choice == '2':
        model = load_model()
        print("Model loaded. Enter image path for prediction.")
        image_path = input("Image path: ")
        race_mapping = {
            0: 'East Asian',
            1: 'Indian',
            2: 'Black',
            3: 'White',
            4: 'Middle Eastern',
            5: 'Latino_Hispanic',
            6: 'Southeast Asian'
        }

        predict_image(model, image_path, race_mapping)
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
