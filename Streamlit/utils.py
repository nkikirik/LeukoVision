import torch
import numpy as np
import cv2

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