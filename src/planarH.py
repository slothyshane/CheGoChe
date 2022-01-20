import numpy as np
import math
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	num_matches = np.array(x1).shape[0]
	A = np.zeros((2*num_matches, 9))
	for i in range(num_matches):
		A[2*i, :] = [x2[i, 0], x2[i, 1], 1, 0, 0, 0, -1 * x2[i, 0] * x1[i, 0],
				-1 * x2[i, 1] * x1[i, 0], -1 * x1[i, 0]]
		A[2*i + 1, :] = [0, 0, 0, x2[i, 0], x2[i, 1], 1,  -1 * x2[i, 0] * x1[i, 1],
				-1 * x2[i, 1] * x1[i, 1], -1 * x1[i, 1]]
	eig_vals, eig_vect = np.linalg.eig(np.matmul(np.transpose(A),A))
	h = eig_vect[:, np.argmin(eig_vals)]
	H2to1 = h.reshape((3, 3))
	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	mean1 = np.mean(x1, axis=0)
	mean2 = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	xnew1 = np.append(x1, np.ones((x1.shape[0], 1)), axis=1)
	xnew2 = np.append(x2, np.ones((x2.shape[0], 1)), axis=1)
	shift1 = np.eye(3)
	shift2 = np.eye(3)
	shift1[0, 2] = -1 * mean1[0]
	shift1[1, 2] = -1 * mean1[1]
	shift2[0, 2] = -1 * mean2[0]
	shift2[1, 2] = -1 * mean2[1]
	for i in range(xnew1.shape[0]):
		xnew1[i, :] = np.matmul(shift1, xnew1[i,:])
		xnew2[i, :] = np.matmul(shift2, xnew2[i,:])

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	dist1 = np.sqrt(np.multiply(xnew1[:, 0], xnew1[:, 0]) + np.multiply(xnew1[:, 1], xnew1[:, 1]))
	scale1 = np.eye(3)
	scale1[0, 0] = math.sqrt(2) / np.amax(dist1)
	scale1[1, 1] = math.sqrt(2) / np.amax(dist1)
	dist2 = np.sqrt(np.multiply(xnew2[:, 0], xnew2[:, 0]) + np.multiply(xnew2[:, 1], xnew2[:, 1]))
	scale2 = np.eye(3)
	scale2[0, 0] = math.sqrt(2) / np.amax(dist2)
	scale2[1, 1] = math.sqrt(2) / np.amax(dist2)
	for i in range(xnew1.shape[0]):
		xnew1[i, :] = np.matmul(scale1, xnew1[i, :])
		xnew2[i, :] = np.matmul(scale2, xnew2[i, :])

	#Similarity transform 1
	T1 = np.matmul(scale1, shift1)

	#Similarity transform 2
	T2 = np.matmul(scale2, shift2)

	#Compute homography
	H = computeH(xnew1[:, 0:2], xnew2[:, 0:2])

	#Denormalization
	H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), H), T2)

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	num_points = 4
	N = locs1.shape[0]
	fit_scores = np.zeros(max_iters)
	bestH2to1 = None
	inliers = None
	for i in range(max_iters):
		inliers_i = np.zeros(N)
		sample_idx = np.zeros(num_points)
		while np.unique(sample_idx).shape[0] < 4:
			for j in range(num_points):
				sample_idx[j] = np.random.randint(N)
		samples1 = np.zeros((num_points, 2))
		samples2 = np.zeros((num_points, 2))
		for j in range(num_points):
			samples1[j, :] = locs1[round(sample_idx[j]), :]
			samples2[j, :] = locs2[round(sample_idx[j]), :]
		H2to1 = computeH_norm(samples1, samples2)
		for j in range(N):
			x2 = [locs2[j, 0], locs2[j, 1], 1]
			x1 = np.matmul(H2to1, x2)
			x1 = np.divide(x1, x1[2], where=(x1[2] != 0))
			dist = np.sqrt(np.power((x1[0] - locs1[j, 0]), 2) + np.power((x1[1] - locs1[j, 1]), 2))
			if dist < inlier_tol:
				inliers_i[j] = 1
		num_inliers = np.sum(inliers_i)
		fit_scores[i] = num_inliers/N
		if np.argmax(fit_scores) == i:
			bestH2to1 = H2to1
			inliers = inliers_i

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
	mask = np.zeros((template.shape[0], template.shape[1], 3)).astype(np.uint8) + 255

	#Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

	#Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

	#Use mask to combine the warped template and the image
	composite_img = np.where(warped_mask != 0, warped_template, img)

	return composite_img


