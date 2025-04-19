import os
from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import torch


def detect_figures(image_path, output_path=None, device=None):
    """
    Detects figures and pictures in an image and returns their coordinates.

    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save annotated image (if None, won't save)
        device (str, optional): Device to run inference on ('cuda:0' or 'cpu')

    Returns:
        list: List of dictionaries containing coordinates and labels of detected figures
    """
    # Set device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} ({'GPU' if 'cuda' in device else 'CPU'})")

    # Download model weights from huggingface
    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
        filename="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
    )

    # Initialize model
    model = YOLOv10(weights)

    # Predict
    result = model.predict(
        image_path,
        imgsz=1024,
        conf=0.2,
        device=device
    )

    # Get bounding boxes and class indices
    boxes = result[0].boxes.xyxy.cpu().numpy()  # (N,4)
    classes = result[0].boxes.cls.cpu().numpy().astype(int)
    names = result[0].names  # dict(int: str), class names

    # Save annotated image if output_path is provided
    if output_path:
        annotated = result[0].plot(pil=True, line_width=5, font_size=20)
        plt.figure(figsize=(12, 12))
        plt.imshow(annotated)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    # Collect coordinates of figures and pictures
    detected_figures = []
    for i, (xyxy, cls) in enumerate(zip(boxes, classes)):
        label = names[cls]
        if label.lower() in ['figure', 'picture']:
            x1, y1, x2, y2 = map(int, xyxy)
            detected_figures.append({
                'id': i + 1,
                'label': label,
                'coordinates': (x1, y1, x2, y2)
            })
            print(f"Detected {label} {i + 1} at: {x1}, {y1}, {x2}, {y2}")

    return detected_figures


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    # Example usage
    image_path = r"C:\Users\feder\Programmazione\LaTeX\gptpdf_LaTeX\Screenshot.png"
    figures = detect_figures(image_path)

    # Get the base filename without path for LaTeX
    image_filename = os.path.basename(image_path)

    # Print LaTeX code for all detected figures using trim and clip
    print("\nLaTeX code for all detected figures:")
    for fig in figures:
        x1, y1, x2, y2 = fig['coordinates']

        # In LaTeX trim, the order is: left bottom right top
        # We need to calculate bottom from the image height
        with Image.open(image_path) as img:
            img_height = img.height

        # Convert to LaTeX trim parameters (left, bottom, right, top)
        # Note: bottom is measured from the bottom of the image, so we need to convert
        left = x1
        bottom = img_height - y2  # Convert from top-left to bottom-left coordinate system
        right = img.width - x2
        top = y1

        latex_code = f"\\begin{{center}}\n  \\includegraphics[trim={{{left}pt {bottom}pt {right}pt {top}pt}}, clip, width=0.7\\linewidth]{{{image_filename}}}\n\\end{{center}}"
        print(latex_code)