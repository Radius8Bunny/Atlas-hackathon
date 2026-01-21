import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, optimizers, applications

# good logging system implemented for debugging, im going to remove this in the final product
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_advanced_model(num_classes, input_shape=(224, 224, 3)):
    base_model = applications.EfficientNetV2B0(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='swish'),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.math.reduce_sum(loss, axis=1)
    return focal_loss_fixed

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    return mixed_x, mixed_y

def train_pipeline(data_path, epochs=50, batch_size=32):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=0.2, subset="both", seed=1337, 
        image_size=(224, 224), batch_size=batch_size, label_mode='categorical'
    )

    num_species = len(train_ds.class_names)
    logger.info(f"Initialized training for {num_species} species: {train_ds.class_names}")

    train_ds = train_ds.map(lambda x, y: mixup_data(x, y)).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = build_advanced_model(num_species)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
    lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

    
    logger.info("starting phase 1: train the classi. head")
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop, lr_schedule])

    logger.info("starting phase 2: the entire network fine tuning")
    model.layers[0].trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5), # im using a very slow learning rate for max retention
        loss=focal_loss(), 
        metrics=['accuracy']
    )
    
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop, lr_schedule])

    model.save("final_brain.keras")

if __name__ == "__main__":
    train_pipeline("Maharashtra_Spectrograms")