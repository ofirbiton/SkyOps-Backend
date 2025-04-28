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


# In utils/visualization.py

from shared import dependencies as dep

def get_original_image_plot(original_image):
    fig = dep.plt.figure()
    dep.plt.imshow(original_image, cmap='gray')
    dep.plt.title("Original Image")
    return fig

def get_binary_image_plot(binary_image):
    fig = dep.plt.figure()
    dep.plt.imshow(binary_image, cmap='gray')
    dep.plt.title("Binary Image (Before Filtering)")
    return fig

def get_skeleton_image_plot(skeleton_image):
    fig = dep.plt.figure()
    dep.plt.imshow(skeleton_image, cmap='gray')
    dep.plt.title("Skeletonized Image")
    return fig

def get_refined_skeleton_plot(refined_skeleton):
    fig = dep.plt.figure()
    dep.plt.imshow(refined_skeleton, cmap='gray')
    dep.plt.title("Skeleton Without Deadends")
    return fig

def get_merged_image_plot(merged_image):
    fig = dep.plt.figure()
    dep.plt.imshow(merged_image)
    dep.plt.title("Merged Image (Buildings + Skeleton)")
    return fig

def get_junctions_highlighted_plot(junctions_highlighted):
    fig = dep.plt.figure()
    dep.plt.imshow(junctions_highlighted)
    dep.plt.title("Highlighted Junctions")
    return fig

def get_refined_junctions_plot(refined_junctions):
    fig = dep.plt.figure()
    dep.plt.imshow(refined_junctions)
    dep.plt.title("Refined Yellow Junctions")
    return fig

def get_final_image_plot(final_image):
    fig = dep.plt.figure()
    dep.plt.imshow(final_image)
    dep.plt.title("Final Image with Path")
    return fig
