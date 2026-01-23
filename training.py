import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, optimizers, applications

print("Setting up model and data...")

def get_model(n_classes, inp_shape=(224, 224, 3)):
    base = applications.EfficientNetV2B0(input_shape=inp_shape, include_top=False, weights='imagenet')
    base.trainable = False 

    m = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='swish'),
        layers.Dense(n_classes, activation='softmax', dtype='float32')
    ])
    return m

def custom_focal(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        
        ce = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * ce
        return tf.math.reduce_sum(loss, axis=1)
    return loss_fn

def do_mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    
    batch = tf.shape(x)[0]
    idx = tf.random.shuffle(tf.range(batch))

    mx = lam * x + (1 - lam) * tf.gather(x, idx)
    my = lam * y + (1 - lam) * tf.gather(y, idx)
    return mx, my

def train(path, epochs=50, bs=32):
    tr_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        path, validation_split=0.2, subset="both", seed=1337, 
        image_size=(224, 224), batch_size=bs, label_mode='categorical'
    )

    classes = tr_ds.class_names
    num_c = len(classes)
    print(f"Found {num_c} classes: {classes}")

    tr_ds = tr_ds.map(lambda x, y: do_mixup(x, y)).prefetch(tf.data.AUTOTUNE)

    model = get_model(num_c)
    
    f_loss = custom_focal()
    
    stops = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

    print("Starting phase 1...")
    model.compile(optimizer=optimizers.Adam(1e-3), loss=f_loss, metrics=['accuracy'])
    model.fit(tr_ds, validation_data=val_ds, epochs=15, callbacks=[stops, reduce_lr])

    print("Starting phase 2 (fine tuning)...")
    model.layers[0].trainable = True
    model.compile(optimizer=optimizers.Adam(1e-5), loss=f_loss, metrics=['accuracy'])
    
    model.fit(tr_ds, validation_data=val_ds, epochs=epochs, callbacks=[stops, reduce_lr])

    model.save("final_brain_v3.keras")
    print("Done. Model saved.")

train("Maharashtra_Spectrograms")