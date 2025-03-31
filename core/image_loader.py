from shared import dependencies as dep

def estimate_thresholds_by_background(img_gray):
    """
    קובע ספים בהתאם לגוון הרקע: צהבהב או אפור.
    """
    h, w = img_gray.shape
    margin = 10

    top = img_gray[:margin, :]
    bottom = img_gray[-margin:, :]
    left = img_gray[:, :margin]
    right = img_gray[:, -margin:]

    sample = dep.np.concatenate([
        top.flatten(),
        bottom.flatten(),
        left.flatten(),
        right.flatten()
    ])

    avg_background = dep.np.mean(sample)
    if avg_background > 235:
        return 245, 247  # רקע צהבהב
    else:
        return 244, 249  # רקע אפור



def load_and_preprocess_image(file_path, lower_threshold=None, upper_threshold=None, min_area=50):
    """
    טוענת תמונה בגווני אפור, מבצעת בינריזציה לפי ערכי סף, ומסננת אזורים קטנים.
    מחזירה: (תמונה מקורית, תמונה בינארית מסוננת)
    """
    img = dep.cv2.imread(file_path, dep.cv2.IMREAD_UNCHANGED)

    # המרה לאפור לפי פורמט הקלט
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_gray = dep.cv2.cvtColor(img, dep.cv2.COLOR_BGR2GRAY)
    elif len(img.shape) > 2 and img.shape[2] == 4:
        img_gray = dep.cv2.cvtColor(img, dep.cv2.COLOR_BGRA2GRAY)
    else:
        img_gray = img

    # אם לא נמסרו ספים – חשב אותם אוטומטית
    if lower_threshold is None or upper_threshold is None:
        lower_threshold, upper_threshold = estimate_thresholds_by_background(img_gray)

    # בינריזציה לפי טווח סף
    img_binary = ((img_gray >= lower_threshold) & (img_gray <= upper_threshold)).astype(dep.np.uint8)

    # סינון לפי שטח אזור
    labeled_image, num_labels = dep.label(img_binary, connectivity=2, return_num=True)
    regions = dep.regionprops(labeled_image)
    large_labels = [region.label for region in regions if region.area >= min_area]
    img_filtered = dep.np.isin(labeled_image, large_labels).astype(dep.np.uint8)

    return img, img_filtered