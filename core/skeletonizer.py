from shared import dependencies as dep

def skeletonize_image(binary_image):
    """
    יוצר שלד מתמונה בינארית (הופך צבעים, עושה skeletonize ומנקה גבולות).
    """
    img_inverted = 1 - binary_image
    skeleton_image = dep.skeletonize(img_inverted).astype(dep.np.uint8)
    skeleton_image[0, :] = skeleton_image[-1, :] = skeleton_image[:, 0] = skeleton_image[:, -1] = 0
    return skeleton_image

def detect_deadends(skeleton_image):
    """
    מזהה פיקסלים שהם קצוות מתים (רק שכן אחד).
    """
    neighbors_kernel = dep.np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=dep.np.uint8)
    neighbor_counts = dep.cv2.filter2D(skeleton_image.astype(dep.np.uint8), -1, neighbors_kernel)
    return ((skeleton_image == 1) & (neighbor_counts == 1)).astype(dep.np.uint8)

def remove_deadends(skeleton_image):
    """
    מסיר פיקסלים קצוות מתים מהשלד, בלולאה עד שאין עוד.
    """
    refined_skeleton = skeleton_image.copy()
    while True:
        deadends = detect_deadends(refined_skeleton)
        if dep.np.count_nonzero(deadends) == 0:
            break
        refined_skeleton[deadends == 1] = 0
    return refined_skeleton

def merge_images(binary_image, refined_skeleton):
    """
    ממזג את התמונה הבינארית (מבנים בלבן) עם השלד (כחול).
    """
    merged_image = dep.np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=dep.np.uint8)
    merged_image[binary_image == 1] = [255, 255, 255]   # מבנים – לבן
    merged_image[refined_skeleton == 1] = [0, 0, 255]   # שלד – כחול
    return merged_image
