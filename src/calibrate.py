import numpy as np
import cv2
from opts import get_opts
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from planarH import computeH_ransac, compositeH

## Calibrate for Homography
opts = get_opts()
unwarped = cv2.imread('../data/goboard8x8.jpg', -1)
unwarped = cv2.resize(unwarped, (opts.gui_size, opts.gui_size))

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, opts.frame_width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, opts.frame_height)
ret, warped = vid.read()
vid.release()

fig, ax = plt.subplots(nrows=1, ncols=2)
im1 = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(im2, cmap='gray')

num_matches = opts.num_matches
matched1 = np.zeros((num_matches, 2))
matched2 = np.zeros((num_matches, 2))
i = 0
while i < num_matches:
    plt.sca(ax[1])
    x1, y1 = plt.ginput(1, timeout=3600, mouse_stop=2)[0]
    x2, y2 = plt.ginput(1, timeout=3600, mouse_stop=2)[0]
    # print(x1, y1, x2, y2)
    con = ConnectionPatch(xyA=(x1,y1), xyB=(x2,y2),
                          coordsA="data", coordsB="data",
                          axesA=ax[0], axesB=ax[1], color="red")
    ax[1].add_artist(con)
    plt.draw()
    matched1[i, :] = [np.round(x1).astype(int), np.round(y1).astype(int)]
    matched2[i, :] = [np.round(x2).astype(int), np.round(y2).astype(int)]
    i = i + 1
H2to1, inliers = computeH_ransac(matched1, matched2, opts)
print(inliers)
dst = cv2.warpPerspective(warped, H2to1, (unwarped.shape[1], unwarped.shape[0]))
cv2.imshow('result', dst)
cv2.waitKey(0)
np.save('H2to1.npy', H2to1)

## Calibrate for ignoring depth shading
img = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = opts.k_means_segments
attempts = 10

ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = label.reshape((img.shape[0], img.shape[1]))
plt.figure()
plt.imshow(result_image)
x, y = plt.ginput(1, timeout=3600, mouse_stop=2)[0]
blank_label = result_image[int(y), int(x)]
placement_pixels = np.array(result_image == blank_label).astype(int)
plt.figure()
plt.imshow(placement_pixels)
np.save('mask.npy', placement_pixels)
plt.show()
