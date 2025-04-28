from shared import dependencies as dep

def highlight_dense_skeleton_nodes(merged_image):
    """
    מזהה נקודות שלד עם לפחות 3 שכנים – מסומן כצומת בצהוב.
    """
    blue_mask = (merged_image[:, :, 0] == 0) &  (merged_image[:, :, 1] == 0) & (merged_image[:, :, 2] == 255)

    binary_skeleton = blue_mask.astype(dep.np.uint8)

    neighbors_kernel = dep.np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=dep.np.uint8)
    neighbor_counts = dep.cv2.filter2D(binary_skeleton, -1, neighbors_kernel)

    # צומת = פיקסל כחול עם 3+ שכנים כחולים
    yellow_nodes = (binary_skeleton == 1) & (neighbor_counts >= 3)
    merged_image[yellow_nodes] = [255, 255, 0]  # צהוב

    return merged_image

def refine_yellow_nodes(junctions_highlighted):
    yellow_mask = ((junctions_highlighted[:, :, 0] == 255) &
                   (junctions_highlighted[:, :, 1] == 255) &
                   (junctions_highlighted[:, :, 2] == 0))
    binary_yellow = yellow_mask.astype(dep.np.uint8)
    labeled_yellows, num_labels = dep.label(binary_yellow, connectivity=2, return_num=True)
    refined_junctions = junctions_highlighted.copy()

    for label_idx in range(1, num_labels + 1):
        region = dep.np.where(labeled_yellows == label_idx)
        selected_idx = dep.np.argmin(region[0])  # בחר את העליון ביותר
        y, x = region[0][selected_idx], region[1][selected_idx]
        refined_junctions[y, x] = [255, 255, 0]  # צהוב

    return refined_junctions
