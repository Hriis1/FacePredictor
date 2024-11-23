from keras.api._v2.keras.applications import ResNet50
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Dense, GlobalAveragePooling2D

def build_model(num_race_classes):
    """Build the multi-output model."""
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)

    # Outputs
    race_output = Dense(num_race_classes, activation='softmax', name='race')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)
    age_output = Dense(1, activation='linear', name='age')(x)

    # Combine into one model
    model = Model(inputs=base_model.input, outputs=[race_output, gender_output, age_output])
    model.compile(
        optimizer='adam',
        loss={
            'race': 'categorical_crossentropy',
            'gender': 'binary_crossentropy',
            'age': 'mean_squared_error'
        },
        metrics={
            'race': 'accuracy',
            'gender': 'accuracy',
            'age': 'mae'
        }
    )
    return model
