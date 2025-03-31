from shared import dependencies as dep


def display_images(original_image, binary_image, skeleton_image, refined_skeleton, merged_image, junctions_highlighted, refined_junctions, final_image):
    dep.plt.figure()
    dep.plt.imshow(original_image, cmap='gray')
    dep.plt.title("Original Image")

    dep.plt.figure()
    dep.plt.imshow(binary_image, cmap='gray')
    dep.plt.title("Binary Image (Before Filtering)")

    dep.plt.figure()
    dep.plt.imshow(skeleton_image, cmap='gray')
    dep.plt.title("Skeletonized Image")

    dep.plt.figure()
    dep.plt.imshow(refined_skeleton, cmap='gray')
    dep.plt.title("Skeleton Without Deadends")

    dep.plt.figure()
    dep.plt.imshow(merged_image)
    dep.plt.title("Merged Image (Buildings + Skeleton)")

    dep.plt.figure()
    dep.plt.imshow(junctions_highlighted)
    dep.plt.title("Highlighted Junctions")

    dep.plt.figure()
    dep.plt.imshow(refined_junctions)
    dep.plt.title("Refined Yellow Junctions")

    dep.plt.figure()
    dep.plt.imshow(final_image)
    dep.plt.title("Final Image with Path")

    dep.plt.tight_layout()
    dep.plt.show()
