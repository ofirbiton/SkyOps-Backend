from shared import dependencies as dep

# Creates a skeleton from a binary image (inverts, skeletonizes, cleans borders)
def skeletonize_image(binary_image):
    img_inverted = 1 - binary_image
    skeleton_image = dep.skeletonize(img_inverted).astype(dep.np.uint8)
    skeleton_image[0, :] = skeleton_image[-1, :] = skeleton_image[:, 0] = skeleton_image[:, -1] = 0
    return skeleton_image

# Detects dead-end pixels (pixels with only one neighbor) in the skeleton
def detect_deadends(skeleton_image):
    neighbors_kernel = dep.np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=dep.np.uint8)
    neighbor_counts = dep.cv2.filter2D(skeleton_image.astype(dep.np.uint8), -1, neighbors_kernel)
    return ((skeleton_image == 1) & (neighbor_counts == 1)).astype(dep.np.uint8)

# Removes dead-end pixels from the skeleton in a loop until none remain
def remove_deadends(skeleton_image):
    refined_skeleton = skeleton_image.copy()
    while True:
        deadends = detect_deadends(refined_skeleton)
        if dep.np.count_nonzero(deadends) == 0:
            break
        refined_skeleton[deadends == 1] = 0
    return refined_skeleton

# Merges the binary image (buildings in white) with the skeleton (blue)
def merge_images(binary_image, refined_skeleton):
    merged_image = dep.np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=dep.np.uint8)
    merged_image[binary_image == 1] = [255, 255, 255]
    merged_image[refined_skeleton == 1] = [0, 0, 255]
    return merged_image
