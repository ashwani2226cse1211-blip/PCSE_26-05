import tensorflow as tf

# Apne train kiye hue model ko load karein
model = tf.keras.models.load_model('final_smart_model.h5')

# TFLite converter banayein
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- YEH DO NAYI LINES ERROR KO FIX KARENGI ---
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # TFLite ke normal ops
  tf.lite.OpsSet.SELECT_TF_OPS   # Zaroorat padne par TensorFlow ke ops bhi use karo
]
converter._experimental_lower_tensor_list_ops = False
# -----------------------------------------------

# Ab model ko convert karein
tflite_model = converter.convert()

# Nayi .tflite file ko save karein
with open('final_smart_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model 'final_smart_model.tflite' naam se safaltapoorvak convert ho gaya hai.")