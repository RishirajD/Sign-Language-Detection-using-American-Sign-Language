from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load InceptionV3 without the top layer (pretrained on ImageNet)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom top layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(26, activation='softmax')(x)  # 26 classes for A-Z

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model to retain the pretrained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear images
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Flip horizontally
    fill_mode='nearest'  # Filling pixels
)

# No augmentation for the validation/test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Create train and test data generators
train_generator = train_datagen.flow_from_directory(
    'D:/objdetection_inceptionmodel/data/train',  # Path to your train data
    target_size=(299, 299),  # InceptionV3 expects 299x299 images
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=True  # Shuffle for training
)

validation_generator = test_datagen.flow_from_directory(
    'D:/objdetection_inceptionmodel/data/test',  # Path to your test data
    target_size=(299, 299),  # InceptionV3 expects 299x299 images
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=False  # No need to shuffle test data
)
# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Set the number of epochs based on your requirement
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Steps per epoch
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Validation steps
)

# Save the trained model
model.save('asl_inception_v3_trained.h5')
print("Model trained and saved as 'asl_inception_v3_trained.h5'")
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
