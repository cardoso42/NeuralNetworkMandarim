from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from PIL import Image
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from imblearn.over_sampling import RandomOverSampler

def load_and_flatten_images(folder_path):
    images = []
    labels = []

    for class_label, class_folder in enumerate(os.listdir(folder_path)):
        # print(str(class_label) + " = " + class_folder)
        class_path = os.path.join(folder_path, class_folder)

        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                if not os.path.isdir(image_path):
                    img = np.array(Image.open(image_path)).flatten().astype(int)
                    images.append(img)
                    labels.append(class_folder)

    X = np.array(images)
    y = np.array(labels)

    return X, y

def evaluate(classifier):
    print("Evaluating...")

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
    print(f"Precison: {precision}")

    recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
    print(f"Recall: {recall}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def show_validation_curve(param_range, test_scores_mean, x_label, y_label, title):
    plt.plot(range(len(param_range)), test_scores_mean, label="Test scores", marker='o', linestyle='--')

    plt.xticks(range(len(param_range)), [str(param) for param in param_range], rotation=45)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

def discover_best_one_hidden_layer_size():
    param_range = [8, 342, 684, 1353]
    _, test_scores = validation_curve(
        mlp, X, y, param_name="hidden_layer_sizes", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    show_validation_curve(param_range, test_scores_mean,
                          "Hidden Layer Size", "Accuracy", "Optimization for 1 Hidden Layer Size")
    

def discover_parameters_two_hidden_layers_size():
    param_range = [(8, 8), (8, 16), (16, 8), (12, 8), (12, 12)]
    _, test_scores = validation_curve(
        mlp, X, y, param_name="hidden_layer_sizes", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    show_validation_curve(param_range, test_scores_mean, 
                          "Hidden Layer Size", "Accuracy", "Optimization for 2 Hidden Layers Size")
    
def discover_parameters_activation_function():
    param_range = ['identity', 'logistic', 'tanh', 'relu']
    _, test_scores = validation_curve(
        mlp, X, y, param_name="activation", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    show_validation_curve(param_range, test_scores_mean, 
                          "Activation Function", "Accuracy", "Optimization for Activation Function")
    
def discover_parameters_activation_function():
    param_range = ['identity', 'logistic', 'tanh', 'relu']
    _, test_scores = validation_curve(
        mlp, X, y, param_name="activation", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    show_validation_curve(param_range, test_scores_mean, 
                          "Activation Function", "Accuracy", "Optimization for Activation Function")
    
def discover_parameters_alpha():
    param_range = np.logspace(-6, 3, 10)
    _, test_scores = validation_curve(
        mlp, X, y, param_name="alpha", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    show_validation_curve(param_range, test_scores_mean, 
                          "Alpha Value", "Accuracy", "Optimization for Alpha Value")

X, y = load_and_flatten_images("../../data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Training and evaluating
mlp = MLPClassifier(random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=5000, hidden_layer_sizes=342, activation='logistic', alpha=0.01)
mlp.fit(X_train_resampled, y_train_resampled)
evaluate(mlp)

# # discover_best_one_hidden_layer_size()
# # discover_parameters_two_hidden_layers_size()
# # discover_parameters_activation_function()
# # discover_parameters_alpha()
# # train_sizes, train_scores, test_scores = learning_curve(mlp, X, y, cv=10, n_jobs=-1, train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0])

# # train_scores_mean = np.mean(train_scores, axis=1)
# # test_scores_mean = np.mean(test_scores, axis=1)

# # plt.figure()
# # plt.plot(train_sizes, train_scores_mean, label='Training score')
# # plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
# # plt.xlabel('Training Size')
# # plt.ylabel('Score')
# # plt.title('Learning Curve')
# # plt.legend(loc="best")
# # plt.show()

# Training for real application
# mlp = MLPClassifier(
#     max_iter=5000, 
#     hidden_layer_sizes=342, 
#     activation='logistic', 
#     alpha=0.01
# )
# mlp.fit(X, y)

# mlp_filename = "trained_mlp_model-oversampled.joblib"
# encoder_filename = "label_encoder.joblib"
# joblib.dump(mlp, mlp_filename)

