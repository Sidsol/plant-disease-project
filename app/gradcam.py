"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI.

Generates attention heatmaps showing which regions of the input image
the model focused on for its prediction. This increases user trust
by making the AI's "reasoning" transparent.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import base64
import io
import threading
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Thread lock to prevent concurrent Grad-CAM hook conflicts
_gradcam_lock = threading.Lock()


def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Apply a JET-like colormap to a grayscale array (0-1) without OpenCV/matplotlib.

    Returns an (H, W, 3) uint8 RGB array.
    """
    # Attempt to use matplotlib if available (better quality)
    try:
        from matplotlib import cm

        rgba = cm.jet(gray)  # (H, W, 4)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        pass

    # Pure-numpy JET approximation
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def get_target_layer(model, model_name: str):
    """
    Return the last convolutional layer suitable for Grad-CAM.

    - EfficientNet-B0: backbone.features[-1]
    - CustomCNN: conv5
    """
    if model_name == "efficientnet":
        return model.backbone.features[-1]
    elif model_name == "custom_cnn":
        return model.conv5
    else:
        raise ValueError(f"No Grad-CAM target layer configured for: {model_name}")


def generate_gradcam(
    model,
    model_name: str,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Generate a Grad-CAM heatmap for the given input.

    Args:
        model: The classification model.
        model_name: "efficientnet" or "custom_cnn".
        input_tensor: Preprocessed input tensor (1, C, H, W).
        target_class: Class index to explain. If None, uses predicted class.

    Returns:
        cam: (H_feat, W_feat) numpy array in [0, 1] — the raw attention map.
        logits: (1, num_classes) tensor of raw model output.
    """
    target_layer = get_target_layer(model, model_name)

    activations = None
    gradients = None

    def save_activation(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def save_gradient(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    with _gradcam_lock:
        fwd_handle = target_layer.register_forward_hook(save_activation)
        bwd_handle = target_layer.register_full_backward_hook(save_gradient)

        try:
            # Forward (gradients needed)
            input_tensor = input_tensor.requires_grad_(True)
            logits = model(input_tensor)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            model.zero_grad()
            score = logits[0, target_class]
            score.backward()

            # Compute Grad-CAM
            w = gradients[0].mean(dim=(1, 2))  # (C,)
            cam = (w[:, None, None] * activations[0]).sum(dim=0)  # (H, W)
            cam = torch.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            return cam.cpu().numpy(), logits.detach()
        finally:
            fwd_handle.remove()
            bwd_handle.remove()


def create_heatmap_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay the Grad-CAM heatmap on the original leaf image.

    Args:
        original_image: Original PIL image.
        cam: (H_feat, W_feat) numpy array in [0, 1].
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        PIL Image with the glowing heatmap overlay.
    """
    orig_w, orig_h = original_image.size

    # Resize CAM to original image dimensions
    cam_image = Image.fromarray(np.uint8(cam * 255), mode="L")
    cam_resized = np.array(cam_image.resize((orig_w, orig_h), Image.BILINEAR)) / 255.0

    # Create colored heatmap
    heatmap_rgb = _jet_colormap(cam_resized)

    # Blend with original
    img_array = np.array(original_image).astype(np.float32)
    heat_array = heatmap_rgb.astype(np.float32)
    overlay = (1 - alpha) * img_array + alpha * heat_array
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def heatmap_to_base64(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    max_size: int = 512,
) -> str:
    """
    Generate the heatmap overlay and encode as a base64 JPEG string.

    Args:
        original_image: Original leaf image.
        cam: Grad-CAM array.
        alpha: Overlay opacity.
        max_size: Max dimension for the output image (keeps bandwidth low).

    Returns:
        Base64-encoded JPEG string (no data URI prefix).
    """
    overlay = create_heatmap_overlay(original_image, cam, alpha)

    # Downscale if too large
    w, h = overlay.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        overlay = overlay.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    overlay.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")
