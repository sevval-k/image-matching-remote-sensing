import cv2
import matplotlib.pyplot as plt

def resize_image(img, scale=0.25):
    # Resize the image by the given scale
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height))


# Read the two images
img1 = cv2.imread("foto 1.png", cv2.IMREAD_COLOR)  # sorgu goruntusu
img2 = cv2.imread("foto 2.png", cv2.IMREAD_COLOR)  # sahne goruntusu

# Goruntuleri 1/4 oranında yeniden boyutlandirma 
img1 = resize_image(img1, scale=0.25)
img2 = resize_image(img2, scale=0.25)

img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# SIFT tespit edicisini olustur
sift = cv2.SIFT_create()

# Tanımlayici ve anahtar noktalari tespit et. 
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Brute-Force Eslestiricisi ile noktalari eslestir
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Eslestirme kalitesine gore mesafeleri sirala
matches = sorted(matches, key=lambda x: x.distance)

# Eslestirmeleri ciz (matches[:30] ile en iyi 30 eslestirmenin cizilmesi saglanmistir.) 
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Eslestirme sonuclarinin gorsellestirilmesi
plt.figure(figsize=(15, 8))
plt.imshow(img_matches)
plt.title('Eslestirme Sonucu')
plt.show()
