import tensorflow as tf
from src.data_preprocessing import load_data
from src.model import build_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))
    
    model.save('model.h5') 

if __name__ == "__main__":
    train_model()
