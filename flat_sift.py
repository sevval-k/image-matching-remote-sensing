import cv2

# Görüntüleri yükleyin
image1 = cv2.imread("foto 1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("foto 2.png", cv2.IMREAD_GRAYSCALE)

# Görüntüleri ön işleme tabi tutun
image1_blurred = cv2.GaussianBlur(image1, (5, 5), 0)
image2_blurred = cv2.GaussianBlur(image2, (5, 5), 0)

# SIFT algorimasını başlatın
sift = cv2.SIFT_create()

# Özellik noktalarını ve descriptor'ları tespit edin
keypoints1, descriptors1 = sift.detectAndCompute(image1_blurred, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2_blurred, None)

# Özellik eşleştirmesi için FLANN-based matcher kullanın
index_params = dict(algorithm=1, trees=10)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Deskriptorları karşılaştırın
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Eşleşmeleri filtrele: sadece en iyi eşleşmeleri alın
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # Lowe's ratio test
        good_matches.append(m)

# En iyi 50 eşleşmeyi al
best_matches = sorted(good_matches, key=lambda x: x.distance)[:40]

# Eşleşmeleri görselleştir
result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Pencereyi normal boyutlandırılabilir hale getir
cv2.namedWindow("Flat Sift Sonucu", cv2.WINDOW_NORMAL)

# Sonuçları görüntüle
cv2.imshow("Flat Sift Sonucu", result_image)

# "s" tuşuna basılınca görüntüyü kaydet
key = cv2.waitKey(0) & 0xFF
if key == ord('s'):  # Kaydetme tuşu
    cv2.imwrite("matches_result.png", result_image)
    print("Görüntü kaydedildi: matches_result.png")

cv2.destroyAllWindows()
