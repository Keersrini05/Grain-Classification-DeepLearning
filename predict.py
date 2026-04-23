import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
model = load_model("grain_classifier_model.h5")

# Class labels (change if needed)
labels = ["Black_gram", "Chenna", "Dhal", "Green_gram", "Wheat"]

# Load image
img_path = "test.jpg"   # put your test image name here
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)[0]
predicted_class = labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Display
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()