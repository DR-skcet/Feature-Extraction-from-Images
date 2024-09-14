import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import easyocr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up OCR reader
reader = easyocr.Reader(['en'])

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to your dataset folder
DATASET_FOLDER = r'C:\Users\91934\Downloads\66e31d6ee96cd_student_resource_3\student_resource 3\images'

def preprocess_image(image_path, img_size=(224, 224)):
    if not os.path.isfile(image_path):
        print(f"Error: File not found - {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read file - {image_path}")
        return None
    
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    return img

def extract_text(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

def create_multi_output_model(img_size=(224, 224, 3)):
    input_layer = Input(shape=img_size)
    
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    weight_output = Dense(1, name='weight')(x)
    volume_output = Dense(1, name='volume')(x)
    dimension_output = Dense(1, name='dimensions')(x)
    
    model = Model(inputs=input_layer, outputs=[weight_output, volume_output, dimension_output])
    model.compile(
        optimizer=Adam(),
        loss='mse',
        metrics={
            'weight': ['mae'],
            'volume': ['mae'],
            'dimensions': ['mae']
        }
    )
    
    return model

def load_data(dataset):
    images = []
    weights = []
    volumes = []
    dimensions = []

    for data in dataset:
        img = preprocess_image(data['image_path'])
        if img is None:
            continue
        images.append(img)
        weights.append(data['weight'])
        volumes.append(data['volume'])
        dimensions.append(data['dimensions'])
    
    images = np.array(images)
    weights = np.array(weights)
    volumes = np.array(volumes)
    dimensions = np.array(dimensions)

    return images, weights, volumes, dimensions

def generate_predictions(model, image_paths):
    predictions = []
    
    for path in image_paths:
        img = preprocess_image(path)
        if img is None:
            predictions.append({"index": os.path.basename(path).split('.')[0], "prediction": ""})
            continue
        
        y_pred_weight, y_pred_volume, y_pred_dimensions = model.predict(np.expand_dims(img, axis=0))
        
        weight = y_pred_weight[0][0]
        volume = y_pred_volume[0][0]
        dimension = y_pred_dimensions[0][0]
        
        # Example conversion to string format
        weight_str = f"{weight:.2f} gram"  # Adjust unit based on actual prediction and units
        volume_str = f"{volume:.2f} litre"  # Adjust unit based on actual prediction and units
        dimension_str = f"{dimension:.2f} cm"  # Adjust unit based on actual prediction and units
        
        prediction = f"{weight_str}, {volume_str}, {dimension_str}"  # Concatenate predictions
        
        predictions.append({"index": os.path.basename(path).split('.')[0], "prediction": prediction})
    
    return predictions

def save_predictions(predictions, output_path):
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)

# Example dataset
dataset = [
    {'image_path': os.path.join(DATASET_FOLDER, 'image1.jpg'), 'weight': 10, 'volume': 20, 'dimensions': 15},
    {'image_path': os.path.join(DATASET_FOLDER, 'image2.jpg'), 'weight': 8, 'volume': 25, 'dimensions': 12},
    # Add more image paths...
]

images, weights, volumes, dimensions = load_data(dataset)

if len(images) > 1:
    X_train, X_val, y_weight_train, y_weight_val, y_volume_train, y_volume_val, y_dimension_train, y_dimension_val = train_test_split(
        images, weights, volumes, dimensions, test_size=0.1, random_state=42
    )
    
    model = create_multi_output_model()

    history = model.fit(
        X_train, 
        [y_weight_train, y_volume_train, y_dimension_train], 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_val, [y_weight_val, y_volume_val, y_dimension_val])
    )

    y_pred_weight, y_pred_volume, y_pred_dimensions = model.predict(X_val)

    mae_weight = mean_absolute_error(y_weight_val, y_pred_weight)
    mae_volume = mean_absolute_error(y_volume_val, y_pred_volume)
    mae_dimension = mean_absolute_error(y_dimension_val, y_pred_dimensions)

    mse_weight = mean_squared_error(y_weight_val, y_pred_weight)
    mse_volume = mean_squared_error(y_volume_val, y_pred_volume)
    mse_dimension = mean_squared_error(y_dimension_val, y_pred_dimensions)

    print(f"MAE for weight: {mae_weight}")
    print(f"MAE for volume: {mae_volume}")
    print(f"MAE for dimensions: {mae_dimension}")

    print(f"MSE for weight: {mse_weight}")
    print(f"MSE for volume: {mse_volume}")
    print(f"MSE for dimensions: {mse_dimension}")

else:
    print("Not enough data to split into training and validation sets.")

# Load your test image paths
test_image_paths = [os.path.join(DATASET_FOLDER, 'image1.jpg'), os.path.join(DATASET_FOLDER, 'image2.jpg')]

# Generate predictions
predictions = generate_predictions(model, test_image_paths)

# Save to CSV
save_predictions(predictions, 'test_out.csv')
