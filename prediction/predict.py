import numpy as np

def predict(model, scaler, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    input_std = scaler.transform(input_array)
    prediction = model.predict(input_std)
    predicted_label = np.argmax(prediction)

    print("Prediction:", prediction)

    if predicted_label == 0:
        print("The tumor is Malignant")
    else:
        print("The tumor is Benign")

    return predicted_label
