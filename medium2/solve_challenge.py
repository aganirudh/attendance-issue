import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = 'fashion_classifier (1).h5'
CLASS_NAMES_PATH = 'class_names (1).txt'
# Updated to match your folder structure
IMAGE_PATH = 'images/class_8_img_18.png' 

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None

def get_name(class_id, names):
    if names and class_id < len(names):
        return names[class_id].split(': ')[-1]
    return str(class_id)

def create_adversarial_pattern(input_image, input_label, model):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        
        # --- FIX IS HERE ---
        # We reshape the label to (1,) so it matches the batch size of the prediction
        label_tensor = tf.reshape(input_label, (1,))
        loss = loss_object(label_tensor, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def solve():
    print("\n--- ⚔️  STARTING ATTACK ⚔️  ---")

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    # 2. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return
        
    print(f"Loading target: {IMAGE_PATH}")
    img = tf.keras.utils.load_img(IMAGE_PATH, color_mode="grayscale", target_size=(28, 28))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    input_tensor = tf.convert_to_tensor(img_array)

    # 3. Initial Prediction
    initial_preds = model.predict(input_tensor, verbose=0)
    initial_label = np.argmax(initial_preds)
    initial_name = get_name(initial_label, class_names)
    print(f"\n[BEFORE] The AI sees: '{initial_name}'")

    # 4. Attack
    perturbations = create_adversarial_pattern(input_tensor, initial_label, model)
    
    # Epsilon = noise amount. 0.1 is usually good.
    epsilon = 0.3
    adv_x = input_tensor + (epsilon * perturbations)
    adv_x = tf.clip_by_value(adv_x, 0, 1)

    # 5. Result
    new_preds = model.predict(adv_x, verbose=0)
    new_label = np.argmax(new_preds)
    new_name = get_name(new_label, class_names)

    print(f"[AFTER]  The AI sees: '{new_name}'")

    if new_label != initial_label:
        print("\n✅ SUCCESS: Model Fooled!")
        print(f"You turned a {initial_name} into a {new_name}.")
    else:
        print("\n❌ FAILED: Prediction didn't change. Try increasing epsilon to 0.2")

if __name__ == "__main__":
    solve()
