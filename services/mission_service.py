import os
import json
from flask import request
from core import metrics
from shared import dependencies as dep
from core.image_loader import load_and_preprocess_image
from core.skeletonizer import skeletonize_image, remove_deadends, merge_images
from core.junction_detector import highlight_dense_skeleton_nodes, refine_yellow_nodes
from core.graph_builder import connect_yellow_junctions, add_point_to_graph, line_intersects_building
from core.pathfinder import dijkstra, optimize_path
from services.mission_utils import error_response, save_uploaded_file, parse_coord, find_color_pixel
from services.mission_io import generate_and_respond_path, handle_direct_route

GREEN = (0, 255, 0)
RED = (0, 0, 255)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'uploads')


def handle_direct_route(takeoff_pixel, landing_pixel, building_mask, satellite_path, X_top_left, Y_top_left, X_bottom_right, Y_bottom_right, original_image):
    mask_image = (building_mask * 255).astype(dep.np.uint8)
    mask_image = dep.cv2.cvtColor(mask_image, dep.cv2.COLOR_GRAY2BGR)
    dep.cv2.line(mask_image, takeoff_pixel, landing_pixel, GREEN, 2)
    path_int = [takeoff_pixel, landing_pixel]
    metrics.print_all_metrics(
        path_raw=path_int,
        path_opt=path_int,
        takeoff=takeoff_pixel,
        landing=landing_pixel,
        building_mask=building_mask,
        image_size=(mask_image.shape[1], mask_image.shape[0])
    )
    return generate_and_respond_path(
        path_int=path_int,
        original_image=mask_image,
        satellite_path=satellite_path,
        takeoff_pixel=takeoff_pixel,
        landing_pixel=landing_pixel,
        X_top_left=X_top_left,
        Y_top_left=Y_top_left,
        X_bottom_right=X_bottom_right,
        Y_bottom_right=Y_bottom_right
    )


def create_mission(request):
    try:
        if "buildings_image" not in request.files or "satellite_image" not in request.files:
            return error_response("Missing files: buildings_image and/or satellite_image.")
        buildings_path = save_uploaded_file(request.files["buildings_image"], UPLOAD_FOLDER)
        satellite_path = save_uploaded_file(request.files["satellite_image"], UPLOAD_FOLDER)
        top_left_coord_str = request.form.get("top_left_coord")
        bottom_right_coord_str = request.form.get("bottom_right_coord")
        if not (top_left_coord_str and bottom_right_coord_str):
            return error_response("Missing top_left_coord or bottom_right_coord")
        X_top_left, Y_top_left = parse_coord(top_left_coord_str)
        X_bottom_right, Y_bottom_right = parse_coord(bottom_right_coord_str)
        original_image, binary_image = load_and_preprocess_image(buildings_path)
        if len(original_image.shape) == 3 and original_image.shape[2] == 4:
            original_image = dep.cv2.cvtColor(original_image, dep.cv2.COLOR_BGRA2BGR)
        takeoff_pixel = find_color_pixel(original_image, GREEN)
        landing_pixel = find_color_pixel(original_image, RED)
        if takeoff_pixel is None or landing_pixel is None:
            return error_response("Could not find takeoff and/or landing pixels.")
        building_mask = (binary_image == 1).astype(dep.np.uint8)
        if not line_intersects_building(takeoff_pixel[0], takeoff_pixel[1], landing_pixel[0], landing_pixel[1], building_mask):
            return handle_direct_route(takeoff_pixel, landing_pixel, building_mask, satellite_path, X_top_left, Y_top_left, X_bottom_right, Y_bottom_right, original_image)
        skeleton_image = skeletonize_image(binary_image)
        refined_skeleton = remove_deadends(skeleton_image)
        merged_image = merge_images(binary_image, refined_skeleton)
        junctions_highlighted = highlight_dense_skeleton_nodes(merged_image)
        refined_junctions = refine_yellow_nodes(junctions_highlighted)
        yellow_mask = ((refined_junctions[:, :, 0] == 255) &
                       (refined_junctions[:, :, 1] == 255) &
                       (refined_junctions[:, :, 2] == 0))
        skeleton_mask = ((merged_image[:, :, 0] == 0) &
                         (merged_image[:, :, 1] == 0) &
                         (merged_image[:, :, 2] == 255))
        final_image, node_list, adjacency_dict = connect_yellow_junctions(merged_image, yellow_mask, skeleton_mask)
        start_node = takeoff_pixel
        end_node = landing_pixel
        res_start = add_point_to_graph(start_node, adjacency_dict, building_mask, final_image, ignore_building=True)
        if not res_start:
            return error_response("Could not connect takeoff node to the graph.")
        res_end = add_point_to_graph(end_node, adjacency_dict, building_mask, final_image, ignore_building=True)
        if not res_end:
            return error_response("Could not connect landing node to the graph.")
        if len(node_list) < 3:
            if not line_intersects_building(takeoff_pixel[0], takeoff_pixel[1], landing_pixel[0], landing_pixel[1], building_mask):
                return handle_direct_route(takeoff_pixel, landing_pixel, building_mask, satellite_path, X_top_left, Y_top_left, X_bottom_right, Y_bottom_right, final_image)
            else:
                return error_response("No path found (only 2 points, and direct line blocked).", 404)
        path = dijkstra(start_node, end_node, adjacency_dict)
        if path is None:
            return error_response("No path found.", 404)
        path_raw = path.copy()
        path = optimize_path(path, building_mask)
        path_int = [(int(x), int(y)) for (x, y) in path]
        metrics.print_all_metrics(
            path_raw=path_raw,
            path_opt=path_int,
            takeoff=takeoff_pixel,
            landing=landing_pixel,
            building_mask=building_mask,
            image_size=(final_image.shape[1], final_image.shape[0])
        )
        return generate_and_respond_path(
            path_int=path_int,
            original_image=final_image,
            satellite_path=satellite_path,
            takeoff_pixel=takeoff_pixel,
            landing_pixel=landing_pixel,
            X_top_left=X_top_left,
            Y_top_left=Y_top_left,
            X_bottom_right=X_bottom_right,
            Y_bottom_right=Y_bottom_right
        )
    except Exception as e:
        return error_response(f"Error: {str(e)}", 500)
