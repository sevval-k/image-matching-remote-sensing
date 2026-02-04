import cv2
import numpy as np
import sys

# Load images
image1 = cv2.imread("foto 1.png")
image2 = cv2.imread("foto 2.png")

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("One or both images could not be loaded.")
    sys.exit()

# Apply Gaussian blur to the images
image1_blurred = cv2.GaussianBlur(image1, (5, 5), 0)
image2_blurred = cv2.GaussianBlur(image2, (5, 5), 0)

# Initialize AKAZE algorithm
akaze = cv2.AKAZE_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = akaze.detectAndCompute(image1_blurred, None)
keypoints2, descriptors2 = akaze.detectAndCompute(image2_blurred, None)

# Convert descriptors to np.float32 format
descriptors1 = np.float32(descriptors1)
descriptors2 = np.float32(descriptors2)

# Set FLANN-based matcher parameters
index_params = dict(algorithm=1, trees=10)  # LSH algorithm
search_params = dict(checks=80)  # Fast search

# Create FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Compare descriptors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filter matches: only keep the best matches (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.732424 * n.distance:  # Lowe's ratio test
        good_matches.append(m)

# Select matching points
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Check if we have enough matches for homography
if len(good_matches) < 4:
    print("Not enough matches.")
    sys.exit()

# Visualize initial good matches
print(f"Number of good matches (initial filtering): {len(good_matches)}")
result_image_initial = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Initial Good Matches", result_image_initial)
cv2.waitKey(0)

# Compute homography using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Select only the good matches after RANSAC
good_matches_ransac = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]

# Print the number of good matches after RANSAC
print(f"Number of good matches (after RANSAC): {len(good_matches_ransac)}")

# Visualize the results
result_image_ransac = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches_ransac, None, flags=cv2.DrawMatchesFlags_DEFAULT)

# Show the RANSAC result
cv2.imshow("Good Matches after RANSAC", result_image_ransac)

# Apply k-means clustering on the matching points (src_pts or dst_pts)
k = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Perform k-means clustering
points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches_ransac])
ret, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Visualize clusters
clustered_image = image1.copy()
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

for i, point in enumerate(points):
    cluster_idx = labels[i][0]
    center = tuple(map(int, centers[cluster_idx]))
    pt = tuple(map(int, point))
    cv2.circle(clustered_image, pt, 5, colors[cluster_idx % len(colors)], -1)
    cv2.line(clustered_image, pt, center, colors[cluster_idx % len(colors)], 1)

# Display clustered image
cv2.imshow("Clusters in Image1", clustered_image)

key = cv2.waitKey(0) & 0xFF
if key == ord('s'):  # Save the image if 's' is pressed
    cv2.imwrite("ransac_matches_clusters.png", clustered_image)
    print("Clustered image saved: ransac_matches_clusters.png")

cv2.destroyAllWindows()