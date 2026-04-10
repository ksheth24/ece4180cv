"""
is_majority_blue.py
-------------------
Takes an image path as input and returns True if the majority
of the image is blue, False otherwise.

Usage:
    python is_majority_blue.py <image_path>
    
    or start Flask server:
    python main.py server

Requirements:
    pip install Pillow numpy flask
"""

import sys
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)


def is_majority_blue(image_path: str, threshold: float = 0.5) -> bool:
    """
    Returns True if more than `threshold` (default 50%) of the image's
    pixels are classified as "blue".

    A pixel is considered blue when:
      - The blue channel is the dominant channel (B > R and B > G)
      - The blue channel value is meaningfully strong (B > 60)
      - The pixel is not too dark/black (B > 40)
      - The pixel is not too white/grey (blue dominance ratio B/(R+G+B) > 0.38)

    Args:
        image_path: Path to the image file.
        threshold:  Fraction of pixels that must be blue (0.0–1.0).

    Returns:
        bool — True if majority blue, False otherwise.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.float32)  # shape: (H, W, 3)

    R = pixels[:, :, 0]
    G = pixels[:, :, 1]
    B = pixels[:, :, 2]

    total = R + G + B + 1e-6  # avoid division by zero

    blue_mask = (
        (B > R) &                     # blue dominates red
        (B > G) &                     # blue dominates green
        (B > 60) &                    # not too dark
        ((B / total) > 0.38)          # blue makes up >38% of the pixel's colour
    )

    blue_fraction = blue_mask.sum() / blue_mask.size
    result = blue_fraction > threshold

    print(f"Blue pixel fraction : {blue_fraction:.2%}")
    print(f"Threshold           : {threshold:.2%}")
    print(f"Majority blue?      : {result}")

    return result


@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Flask endpoint to process image data via POST request.
    
    Expects binary image data in the request body.
    Returns JSON with:
        - blue_fraction: percentage of blue pixels
        - threshold: threshold used
        - is_majority_blue: boolean result
        - success: true if processing succeeded
    """
    try:
        # Get binary image data from request body
        image_data = request.get_data()
        
        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Convert binary data to PIL Image
        img = Image.open(BytesIO(image_data)).convert("RGB")
        pixels = np.array(img, dtype=np.float32)  # shape: (H, W, 3)

        R = pixels[:, :, 0]
        G = pixels[:, :, 1]
        B = pixels[:, :, 2]

        total = R + G + B + 1e-6  # avoid division by zero

        blue_mask = (
            (B > R) &                     # blue dominates red
            (B > G) &                     # blue dominates green
            (B > 60) &                    # not too dark
            ((B / total) > 0.38)          # blue makes up >38% of the pixel's colour
        )

        blue_fraction = blue_mask.sum() / blue_mask.size
        threshold = 0.5
        result = blue_fraction > threshold

        return jsonify({
            "success": True,
            "blue_fraction": float(blue_fraction),
            "threshold": threshold,
            "is_majority_blue": bool(result)
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    # Check if running with command-line args (CLI mode)
    if len(sys.argv) >= 2 and sys.argv[1].lower() != "server":
        # CLI mode: python app.py <image_path> [threshold]
        path = sys.argv[1]
        thresh = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.5
        result = is_majority_blue(path, threshold=thresh)
        sys.exit(0 if result else 1)
    else:
        # Server mode (default) - use PORT env var for Render
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV') == 'development'
        print(f"Starting Flask server on port {port}")
        print(f"Send POST requests to http://localhost:{port}/process_image")
        app.run(host='0.0.0.0', port=port, debug=debug)