

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity

random.seed(a=None, version=2)
set_verbosity(INFO)

def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))
    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals





def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    num_selected_labels = len(selected_labels)
    num_rows = (num_selected_labels + 1) // 3 + 1

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Plot original image
    axes[0, 0].imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    # Plot GradCAM for each selected label
    for j, label in enumerate(selected_labels, start=1):
        i = labels.index(label)
        print(f"Generating gradcam for class {label}")
        gradcam = grad_cam(model, preprocessed_input, i, layer_name)
        row, col = divmod(j, 3)
        axes[row, col].imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
        axes[row, col].set_title(f"{label}: p={predictions[0][i]:.3f}")
        axes[row, col].axis('off')

    # Identify top 4 labels with the highest probabilities
    top_4_indices = np.argsort(predictions[0])[::-1][:4]
    top_4_labels = [labels[i] for i in top_4_indices]

    # Print messages for the top 4 labels
    print("Top 4 likely conditions:")
    for i, label in enumerate(top_4_labels, start=1):
        print(f"{i}. The person is likely suffering from {label}.")

    # Hide any empty subplots
    for j in range(num_selected_labels + 1, num_rows * 3):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
