import os
from flask import jsonify

def error_response(message: str, code: int = 400):
    return jsonify({"message": message, "success": False}), code

def save_uploaded_file(file_storage, upload_folder: str) -> str:
    filename = file_storage.filename
    path = os.path.join(upload_folder, filename)
    file_storage.save(path)
    return path

def parse_coord(coord_str):
    try:
        coord_str = coord_str.strip()
        if coord_str.startswith("(") and coord_str.endswith(")"):
            coord_str = coord_str[1:-1]
        x_str, y_str = coord_str.split(",")
        return float(x_str.strip()), float(y_str.strip())
    except Exception as ex:
        raise ValueError(f"Invalid coordinate format: {coord_str}")

def find_color_pixel(image, target_color):
    import numpy as np
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    mask = np.all(image == np.array(target_color, dtype=np.uint8), axis=-1)
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    y, x = coords[0]
    return (int(x), int(y))
