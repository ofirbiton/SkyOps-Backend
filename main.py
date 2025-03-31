#####################
#      backend      #
#####################

import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random

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

@app.route("/api/create-mission", methods=["POST"])
def create_mission():
    try:
        # 1) בדיקה שיש תמונה
        if "image" not in request.files:
            return jsonify({"message": "Missing field: image (file).", "success": False}), 400

        # 2) שמירת התמונה
        image_file = request.files["image"]
        original_filename = image_file.filename
        image_path = os.path.join(UPLOAD_FOLDER, original_filename)
        image_file.save(image_path)

        # 3) קריאת startX, startY, endX, endY
        startX_str = request.form.get("startX")
        startY_str = request.form.get("startY")
        endX_str = request.form.get("endX")
        endY_str = request.form.get("endY")

        if not (startX_str and startY_str and endX_str and endY_str):
            return jsonify({"message": "Missing start/end coordinates", "success": False}), 400

        startX = int(startX_str)
        startY = int(startY_str)
        endX   = int(endX_str)
        endY   = int(endY_str)
        print("Received startPoint:", (startX, startY), "endPoint:", (endX, endY))

        # 4) עיבוד התמונה
        original_image, binary_image = load_and_preprocess_image(image_path)
        # binary_image אמור להכיל מבנים=1, רקע=0

        skeleton_image = skeletonize_image(binary_image)
        refined_skeleton = remove_deadends(skeleton_image)
        merged_image = merge_images(binary_image, refined_skeleton)

        junctions_highlighted = highlight_dense_skeleton_nodes(merged_image)
        refined_junctions = refine_yellow_nodes(junctions_highlighted)

        yellow_mask = (
            (refined_junctions[:, :, 0] == 255) &
            (refined_junctions[:, :, 1] == 255) &
            (refined_junctions[:, :, 2] == 0)
        )
        skeleton_mask = (
            (merged_image[:, :, 0] == 255) & 
            (merged_image[:, :, 1] == 0) & 
            (merged_image[:, :, 2] == 0)
        )

        final_image, node_list, adjacency_dict = connect_yellow_junctions(
            merged_image,
            yellow_mask,
            skeleton_mask
        )

        if len(node_list) < 2:
            return jsonify({
                "message": "Not enough nodes detected to compute a route",
                "success": False
            }), 400

        # 5) יצירת building_mask: מבנה=1, רקע=0
        # הנחה: binary_image[y,x]==1 => מבנה
        building_mask = (binary_image == 1).astype(np.uint8)

        # הוספת נקודת start ו-end לגרף
        start_node = (startX, startY)
        res_start = add_point_to_graph(start_node, adjacency_dict, building_mask)
        if not res_start:
            return jsonify({
                "message": "Could not connect start node to the graph (maybe all lines cross building?)",
                "success": False
            }), 400

        end_node = (endX, endY)
        res_end = add_point_to_graph(end_node, adjacency_dict, building_mask)
        if not res_end:
            return jsonify({
                "message": "Could not connect end node to the graph",
                "success": False
            }), 400

        # 6) הרצת Dijkstra
        path = dijkstra(start_node, end_node, adjacency_dict)
        if path is None:
            return jsonify({"message":"No path found","success":False}), 404

        path_int = [(int(x), int(y)) for (x, y) in path]

        # ציור המסלול בירוק
        for i in range(len(path_int) - 1):
            xA, yA = path_int[i]
            xB, yB = path_int[i+1]
            dep.cv2.line(final_image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # שמירת תמונה
        output_image_filename = "auto_route.png"
        route_image_path = os.path.join(OUTPUT_FOLDER, output_image_filename)
        dep.cv2.imwrite(route_image_path, final_image)

        # שמירת קובץ txt
        coord_filename = "auto_route_coordinates.txt"
        coord_filepath = os.path.join(OUTPUT_FOLDER, coord_filename)
        coords_json = {
            "start_node": [startX, startY],
            "end_node": [endX, endY],
            "route": path_int
        }
        with open(coord_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(coords_json, indent=2))

        route_image_url = f"/static/outputs/{output_image_filename}"
        coords_file_url = f"/static/outputs/{coord_filename}"

        return jsonify({
            "message": "Mission created successfully (with start/end coords + added to graph)",
            "success": True,
            "routeImageUrl": route_image_url,
            "coordinatesFileUrl": coords_file_url
        }), 200

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({"message": f"Error: {str(e)}", "success":False}), 500

if __name__ == "__main__":
    app.run(debug=True)





# def main():
#     file_path = select_image_file()
#     if not file_path:
#         print("לא נבחר קובץ.")
#         return

    # # שלב 1: טעינה וסינון
    # original_image, binary_image = load_and_preprocess_image(file_path)

    # # שלב 2: שלד
    # skeleton_image = skeletonize_image(binary_image)

    # # שלב 3: הסרת קצוות מתים
    # refined_skeleton = remove_deadends(skeleton_image)

    # # שלב 4: מיזוג מבנים ושלד
    # merged_image = merge_images(binary_image, refined_skeleton)

    # # שלב 5: הדגשת צמתים
    # junctions_highlighted = highlight_dense_skeleton_nodes(merged_image)

    # # שלב 6: טיוב צמתים
    # refined_junctions = refine_yellow_nodes(junctions_highlighted)

    # # שלב 7: מסכות צבעים
    # yellow_mask = (refined_junctions[:, :, 0] == 255) & \
    #               (refined_junctions[:, :, 1] == 255) & \
    #               (refined_junctions[:, :, 2] == 0)
    # skeleton_mask = (merged_image[:, :, 0] == 0) & \
    #                 (merged_image[:, :, 1] == 0) & \
    #                 (merged_image[:, :, 2] == 255)

    # # שלב 8: בניית גרף קשרים
    # final_image, node_list, adjacency_dict = connect_yellow_junctions(
    #     merged_image,
    #     yellow_mask,
    #     skeleton_mask
    # )

    # print("\nNode list (x, y):")
    # for node in node_list:
    #     print(node)

    # # שלב 9: קלט מהמשתמש למסלול
    # if node_list:
    #     try:
    #         start_input = input("\nהקלד קואורדינטות התחלה בפורמט 'x y': ")
    #         x1, y1 = map(int, start_input.split())
    #         start_node = (x1, y1)

    #         end_input = input("הקלד קואורדינטות סוף בפורמט 'x y': ")
    #         x2, y2 = map(int, end_input.split())
    #         end_node = (x2, y2)

    #         if start_node not in adjacency_dict:
    #             print(f"\nהצומת {start_node} לא נמצאת ברשימת הצמתים.")
    #         elif end_node not in adjacency_dict:
    #             print(f"\nהצומת {end_node} לא נמצאת ברשימת הצמתים.")
    #         else:
    #             path = dijkstra(start_node, end_node, adjacency_dict)

    #             if path is None:
    #                 print("\nלא נמצא מסלול בין הצמתים שנבחרו.")
    #             else:
    #                 print("\nמסלול קצר ביותר:")
    #                 for p in path:
    #                     print(p)

    #                 # ציור המסלול בירוק
    #                 for i in range(len(path) - 1):
    #                     x1, y1 = path[i]
    #                     x2, y2 = path[i + 1]
    #                     dep.cv2.line(final_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     except ValueError:
    #         print("פורמט קלט לא תקין. נסה שוב.")

#     # שלב 10: תצוגת כל השלבים
#     display_images(
#         original_image,
#         binary_image,
#         skeleton_image,
#         refined_skeleton,
#         merged_image,
#         junctions_highlighted,
#         refined_junctions,
#         final_image
#     )

# if __name__ == "__main__":
#     main()
