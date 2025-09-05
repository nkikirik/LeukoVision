from captum.attr import LayerGradCam
import numpy as np
import cv2

def show_importance_inception(model, input_tensor, target=0, device="cpu"):
    lig = LayerGradCam(model, model._modules.get('Mixed_7c'))
    input_tensor = input_tensor.to(device)
    model.to(device)
    attributions = lig.attribute(inputs=input_tensor, target=target)
    importance = attributions.sum(dim=1).squeeze(0)
    importance = importance.cpu().detach().numpy()
    importance = np.maximum(importance, 0)  
    importance /= np.max(importance)  
    return importance

def show_importance_resnet(model, input_tensor, target=0, device="cpu"):
    lig = LayerGradCam(model, model._modules.get('layer4'))
    input_tensor = input_tensor.to(device)
    model.to(device)
    attributions = lig.attribute(inputs=input_tensor, target=target)
    importance = attributions.sum(dim=1).squeeze(0)
    importance = importance.cpu().detach().numpy()
    importance = np.maximum(importance, 0)  
    importance /= np.max(importance)  
    return importance

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