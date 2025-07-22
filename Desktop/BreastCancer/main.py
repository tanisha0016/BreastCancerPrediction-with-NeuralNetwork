from data.load_data import load_data
from models.models import build_model
from train_model.train import train_model
from prediction.predict import predict

def main():
    X_train_std, X_test_std, Y_train, Y_test, scaler, dataset = load_data()

    model = build_model()
    model = train_model(model, X_train_std, Y_train)

    loss, accuracy = model.evaluate(X_test_std, Y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Try a sample prediction
    input_sample = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,
                    0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,
                    12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

    predict(model, scaler, input_sample)

    model.summary()

if __name__ == '__main__':
    main()
