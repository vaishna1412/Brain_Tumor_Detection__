import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import pickle

# Load dataset
DATA_DIR = 'dataset/'
categories = ['yes', 'no']
data, labels = [], []

for category in categories:
    folder = os.path.join(DATA_DIR, category)
    class_num = categories.index(category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            data.append(img)
            labels.append(class_num)
        except:
            pass

X = np.array(data).reshape(-1, 100, 100, 1) / 255.0
y = np.array(labels)

# Train CNN
y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
cnn_model.save('models/cnn_model.h5')

# Train ML models
X_flat = np.array(data).reshape(len(data), -1) / 255.0
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_flat)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_pca, y, test_size=0.2)

svm = SVC().fit(X_train_ml, y_train_ml)
rf = RandomForestClassifier().fit(X_train_ml, y_train_ml)
knn = KNeighborsClassifier().fit(X_train_ml, y_train_ml)
ensemble = VotingClassifier(estimators=[
    ('svm', svm),
    ('rf', rf),
    ('knn', knn)
], voting='hard')
ensemble.fit(X_train_ml, y_train_ml)

joblib.dump({'svm': svm, 'rf': rf, 'knn': knn, 'ensemble':ensemble, 'pca': pca}, 'models/ml_models.pkl')

print("All models trained and saved.")

# Accuracy Results
#cnn_acc = cnn_model.evaluate(X, y, verbose=0)[1]
svm_acc = accuracy_score(y_test_ml, svm.predict(X_test_ml))
rf_acc = accuracy_score(y_test_ml, rf.predict(X_test_ml))
knn_acc = accuracy_score(y_test_ml, knn.predict(X_test_ml))
ensemble_acc = accuracy_score(y_test_ml, ensemble.predict(X_test_ml))

acc_results = {
    #'CNN': cnn_acc,
    'SVM': svm_acc,
    'RF': rf_acc,
    'KNN': knn_acc,
    'Ensemble': ensemble_acc
}

with open('models/accuracy_scores.pkl', 'wb') as f:
    pickle.dump(acc_results, f)

    print("Training completed with accuracies:")
for model, acc in acc_results.items():
    print(f"{model} Accuracy: {acc:.2f}")

