import torch
import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_heatmap(img_tensor, model, target_layer_name, pred_index=None):   
    # Skip forward to get target layer predictions and activations
    def forward_hook(module, input, output):
        model.features = output

    # Register a hook on the target layer to retrieve its outputs
    hook = model._modules.get(target_layer_name).register_forward_hook(forward_hook)
    
    # Perform a forward pass to obtain model outputs
    output = model(img_tensor)
    
    # Remove hook after getting activations
    hook.remove()

    # If no prediction index is provided, use the one with the highest probability
    if pred_index is None:
        pred_index = output.argmax(dim=1).item()
    
    # Class probability value
    y = output[0, pred_index]
    
    # Skip backwards to get the gradients of the target layer
    model.zero_grad() # Reset gradients
    model.features.retain_grad() # Keep target layer gradients
    y.backward(retain_graph=True) # Calculate gradients by backpropagation

    # Get target layer gradients and activations
    gradients = model.features.grad[0]
    activations = model.features[0]

    # Apply average global pooling on gradients
    pooled_grads = torch.mean(gradients, dim=[1, 2])

    # Weight activations by gradients
    for i in range(len(pooled_grads)):
        activations[i, :, :] *= pooled_grads[i]

    # Calculate the heatmap
    heatmap = torch.mean(activations, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) # Keep only positive values
    heatmap /= np.max(heatmap) # Normalize the heatmap

    return heatmap, pred_index

def make_gradcam_heatmap_keras(img_array, model, last_conv_layer_name='block5_conv3', pred_index=None):
    # build a model that maps inputs -> (last_conv_output, model_output)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        print (predictions.shape)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # gradients of the target class w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)  # shape: (1, H, W, C)

    # global average pooling on gradients -> importance for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # remove batch dim
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    #heatmap = tf.reduce_sum(conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)

    # post-process
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def get_canny_edge(img, threshold1=30, threshold2=80):
    """
    Function to get the canny edge of an image
    Input: img in (H, W, 3), dtype uint8 or float [0,1]
    Output: edge map (H, W, 3), float in [0,1]
    """
    # If float [0,1], convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get the edge (invert edges so white = edge)
    edge = 255 - cv2.Canny(gray, threshold1, threshold2)

    # Convert to 3-channel float [0,1]
    edge = np.stack([edge]*3, axis=-1) / 255.0
    return edge