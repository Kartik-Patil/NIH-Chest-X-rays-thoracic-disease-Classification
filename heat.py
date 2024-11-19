import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import DenseNet121

# Assuming you have the load_image function defined
def load_image(img, image_dir, df, preprocess=True):
    # Implementation of load_image function goes here
    pass

def grad_cam(model, preprocessed_input, class_index, layer_name):
    # Get the output tensor of the specified layer
    layer_output = model.get_layer(layer_name).output
    
    # Get the gradients of the specified class with respect to the output tensor
    grads = K.gradients(model.output[:, class_index], layer_output)[0]
    
    # Compute the mean of the gradients over each feature map
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    # Access the values of the quantities we computed
    iterate = K.function([model.input], [pooled_grads, layer_output[0]])
    pooled_grads_value, layer_output_value = iterate([preprocessed_input])
    
    # Multiply each channel in the feature map array by "how important this channel is" regarding the class
    for i in range(layer_output.shape[-1]):
        layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    # Average the feature map along the channel dimension resulting in a heatmap
    heatmap = np.mean(layer_output_value, axis=-1)
    
    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

def compute_gradcam(model, img, image_dir, df, labels, selected_labels, layer_name='bn'):
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

    # Plot GradCAM for each selected label and generate heatmaps
    heatmaps = []
    for j, label in enumerate(selected_labels, start=1):
        i = labels.index(label)
        print(f"Generating gradcam for class {label}")
        gradcam = grad_cam(model, preprocessed_input, i, layer_name)
        heatmaps.append(gradcam)
        row, col = divmod(j, 3)
        axes[row, col].imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
        axes[row, col].set_title(f"{label}: p={predictions[0][i]:.3f}")
        axes[row, col].axis('off')

    # Hide any empty subplots
    for j in range(num_selected_labels + 1, num_rows * 3):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    return heatmaps

# Example usage
# Assuming 'model' is your pre-trained model
# 'img', 'image_dir', 'df', 'labels', 'selected_labels' need to be replaced with actual data
compute_gradcam(model, img, image_dir, df, labels, selected_labels)
