#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backend server for processing images.
This server receives:
  - a buildings_image (which should include one green pixel for takeoff and one red pixel for landing),
  - a satellite_image,
  - (the diagonal coordinates are fixed as below)

The server processes the buildings image by applying skeletonization, junction detection,
graph construction and pathfinding (using Dijkstra). It then converts the computed path's
pixel coordinates into real-world coordinates (using the fixed diagonal), and overlays the
path on both the buildings and satellite images.
"""

import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from shared import dependencies as dep
from core.image_loader import load_and_preprocess_image
from core.skeletonizer import skeletonize_image, remove_deadends, merge_images
from core.junction_detector import highlight_dense_skeleton_nodes, refine_yellow_nodes
from core.graph_builder import connect_yellow_junctions, add_point_to_graph
from core.pathfinder import dijkstra

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def parse_coord(coord_str):
    """
    Parses a coordinate string in the format "(x, y)" and returns numeric values.
    """
    try:
        coord_str = coord_str.strip()
        if coord_str.startswith("(") and coord_str.endswith(")"):
            coord_str = coord_str[1:-1]
        x_str, y_str = coord_str.split(",")
        return float(x_str.strip()), float(y_str.strip())
    except Exception as ex:
        raise ValueError(f"Invalid coordinate format: {coord_str}")

def find_color_pixel(image, target_color):
    """
    Finds a pixel in the image that exactly matches the target color.
    
    Note: OpenCV loads images in BGR format.
         Example:
            - Takeoff point: Green (BGR: (0, 255, 0))
            - Landing point: Red (BGR: (0, 0, 255)) corresponding to RGB=(255, 0, 0)
    If the image has 4 channels (with alpha), uses only the first three channels.
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    mask = np.all(image == np.array(target_color, dtype=np.uint8), axis=-1)
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    # np.argwhere returns (y, x); convert to (x, y)
    y, x = coords[0]
    return (int(x), int(y))

@app.route("/api/create-mission", methods=["POST"])
def create_mission():
    try:
        print("Received request to create mission.")
        # 1) Check that both images are provided
        if "buildings_image" not in request.files or "satellite_image" not in request.files:
            print("Missing files: buildings_image and/or satellite_image.")
            return jsonify({"message": "Missing files: buildings_image and/or satellite_image.", "success": False}), 400

        # 2) Save the images
        buildings_file = request.files["buildings_image"]
        satellite_file = request.files["satellite_image"]
        

        buildings_filename = buildings_file.filename
        satellite_filename = satellite_file.filename

        buildings_path = os.path.join(UPLOAD_FOLDER, buildings_filename)
        satellite_path = os.path.join(UPLOAD_FOLDER, satellite_filename)

        buildings_file.save(buildings_path)
        satellite_file.save(satellite_path)

        # 3) Use fixed diagonal coordinates for conversion:
       # === במקום החלק שמנסה לקחת מ‑request.files ===
        top_left_coord_str     = request.form.get("top_left_coord")
        bottom_right_coord_str = request.form.get("bottom_right_coord")

        if not (top_left_coord_str and bottom_right_coord_str):
            return jsonify({"message": "Missing top_left_coord or bottom_right_coord",
                            "success": False}), 400

        X_top_left,  Y_top_left  = parse_coord(top_left_coord_str)
        X_bottom_right, Y_bottom_right = parse_coord(bottom_right_coord_str)

        print("Using fixed diagonal coordinates:", (X_top_left, Y_top_left), (X_bottom_right, Y_bottom_right))

        # 4) Load and pre-process the buildings image.
        original_image, binary_image = load_and_preprocess_image(buildings_path)
        if len(original_image.shape) == 3 and original_image.shape[2] == 4:
            original_image = dep.cv2.cvtColor(original_image, dep.cv2.COLOR_BGRA2BGR)

        # 5) Locate takeoff (green) and landing (red) pixels in the buildings image
        takeoff_pixel = find_color_pixel(original_image, (0, 255, 0))
        landing_pixel = find_color_pixel(original_image, (0, 0, 255))
        if takeoff_pixel is None or landing_pixel is None:
            return jsonify({"message": "Could not find takeoff and/or landing pixels.", "success": False}), 400
        print("Detected takeoff pixel:", takeoff_pixel)
        print("Detected landing pixel:", landing_pixel)

        # 6) Process for graph generation.
        skeleton_image = skeletonize_image(binary_image)
        refined_skeleton = remove_deadends(skeleton_image)
        merged_image = merge_images(binary_image, refined_skeleton)

        junctions_highlighted = highlight_dense_skeleton_nodes(merged_image)
        refined_junctions = refine_yellow_nodes(junctions_highlighted)

        # 7) Create color masks.
        yellow_mask = ((refined_junctions[:, :, 0] == 255) &
                       (refined_junctions[:, :, 1] == 255) &
                       (refined_junctions[:, :, 2] == 0))
        skeleton_mask = ((merged_image[:, :, 0] == 0) &
                         (merged_image[:, :, 1] == 0) &
                         (merged_image[:, :, 2] == 255))

        # 8) Build the connectivity graph.
        final_image, node_list, adjacency_dict = connect_yellow_junctions(
            merged_image,
            yellow_mask,
            skeleton_mask
        )

        if len(node_list) < 2:
            return jsonify({"message": "Not enough nodes detected to compute a route.",
                            "success": False}), 400

        # 9) Create building_mask: in the binary image, buildings == 1.
        building_mask = (binary_image == 1).astype(np.uint8)

        # 10) Connect the takeoff and landing points to the graph.
        start_node = takeoff_pixel
        res_start = add_point_to_graph(start_node, adjacency_dict, building_mask, final_image, ignore_building=True)
        if not res_start:
            return jsonify({
                "message": "Could not connect takeoff node to the graph.",
                "success": False
            }), 400

        end_node = landing_pixel
        res_end = add_point_to_graph(end_node, adjacency_dict, building_mask, final_image, ignore_building=True)
        if not res_end:
            return jsonify({
                "message": "Could not connect landing node to the graph.",
                "success": False
            }), 400


        # 11) Run Dijkstra to compute the shortest path.
        path = dijkstra(start_node, end_node, adjacency_dict)
        if path is None:
            return jsonify({"message": "No path found.", "success": False}), 404

        path_int = [(int(x), int(y)) for (x, y) in path]

        # 12) Convert pixel coordinates to real-world coordinates.
        height, width = original_image.shape[:2]
        real_path = []
        for (pixel_x, pixel_y) in path_int:
            real_x = X_top_left + pixel_x * ((X_bottom_right - X_top_left) / width)
            real_y = Y_top_left - pixel_y * ((Y_top_left - Y_bottom_right) / height)
            real_path.append({"x": real_x, "y": real_y})

        real_takeoff = {
            "x": X_top_left + takeoff_pixel[0] * ((X_bottom_right - X_top_left) / width),
            "y": Y_top_left - takeoff_pixel[1] * ((Y_top_left - Y_bottom_right) / height)
        }
        real_landing = {
            "x": X_top_left + landing_pixel[0] * ((X_bottom_right - X_top_left) / width),
            "y": Y_top_left - landing_pixel[1] * ((Y_top_left - Y_bottom_right) / height)
        }

        # 13) Draw the computed path (in green) on the final image.
        for i in range(len(path_int) - 1):
            xA, yA = path_int[i]
            xB, yB = path_int[i+1]
            dep.cv2.line(final_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        rgb_out = dep.cv2.cvtColor(final_image, dep.cv2.COLOR_BGR2RGB)
        output_graph_filename = "auto_route.png"
        route_image_path = os.path.join(OUTPUT_FOLDER, output_graph_filename)
        dep.cv2.imwrite(route_image_path, rgb_out)
        

        # 14) Overlay the computed path on the satellite image.
        satellite_image = dep.cv2.imread(satellite_path)
        if len(satellite_image.shape) == 3 and satellite_image.shape[2] == 4:
            satellite_image = dep.cv2.cvtColor(satellite_image, dep.cv2.COLOR_BGRA2BGR)
        for i in range(len(path_int) - 1):
            pt1 = path_int[i]
            pt2 = path_int[i+1]
            dep.cv2.line(satellite_image, pt1, pt2, (255, 0, 0), 2)
        output_satellite_filename = "mission_satellite.png"
        satellite_output_path = os.path.join(OUTPUT_FOLDER, output_satellite_filename)
        dep.cv2.imwrite(satellite_output_path, satellite_image)

        # 15) Create a text file with real-world coordinates.
        coord_filename = "auto_route_coordinates.txt"
        coord_filepath = os.path.join(OUTPUT_FOLDER, coord_filename)
        coords_json = {
            "takeoff_point": real_takeoff,
            "landing_point": real_landing,
            "path": real_path
        }

        with open(coord_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(coords_json, indent=2))

        # 16) Return a JSON response with links to the output images and coordinate file.
        return jsonify({
            "message": "Mission created successfully (data processed and functions executed correctly)",
            "success": True,
            "routeImageUrl": url_for('static', filename=f"outputs/{output_graph_filename}", _external=True),
            "satelliteImageUrl": url_for('static', filename=f"outputs/{output_satellite_filename}", _external=True),
            "coordinatesFileUrl": url_for('static', filename=f"outputs/{coord_filename}", _external=True)
        }), 200
    

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({"message": f"Error: {str(e)}", "success": False}), 500

if __name__ == "__main__":
    app.run(debug=True)
    
