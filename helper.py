# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import cv2
random.seed(0)

'''
extract_features, correspondance, extract_and_match from A4
'''

def extract_features(image):
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    
    key_point, descriptor = sift.detectAndCompute(grayimg, None)
    return key_point, descriptor


def correspondance(ref_descriptor, test_descriptors):
    #euclidean distance
    eud = np.linalg.norm(ref_descriptor - test_descriptors, axis = 1)
    #sorted index of eculidean distance
    s = np.argsort(eud)
    #closest descriptor matches index
    c1 = s[0]
    #second closest index
    c2 = s[1]
    #threshold = closest1/closest2
    threshold = eud[c1] / eud[c2]
    
    return c1, threshold

#returns matching key points of two images.
def extract_and_match(img1, img2):
    k1, d1 = extract_features(img1) 
    k2, d2 = extract_features(img2) 
    
    #get match of two photos
    match = []
    for i in range(len(d1)):
        j, threshold = correspondance(d2, d1[i])
        #good match when threshold < 0.8
        if threshold < 0.8:
            match.append((k1[i], k2[j]))
    return match

#calcualte Homography
def homography(match):
    
    A = []
    
    for i in range(len(match)):
        kp_match = match[i]
        (x_i, y_i) = kp_match[0].pt
        (x_j, y_j) = kp_match[1].pt 
         
        #matrix from the lecture slide
        a = [[x_i, y_i, 1, 0, 0, 0, -x_j*x_i, -x_j*y_i, -x_j],
            [0, 0, 0, x_i, y_i, 1, -y_j*x_i, -y_j*y_i, -y_j]]
        
        A.extend(a)

    A = np.array(A)
    #calculate eigenvalues & eigenvectors of A.T @ A
    eigen_values, eigen_vectors = np.linalg.eig(A.T @ A)
    
    # return eigenvector with smallest eigenvalue
    min_idx = np.argsort(eigen_values)[0]
    h = eigen_vectors[:, min_idx]
    H = np.reshape(h, (3,3))
    return H


#transform point
def transform(H, p):
    p = np.array([p[0], p[1],1])
    
    w = H @ p
    w = w/w[2]
    
    return (w[0], w[1])

#calculate number of inliners
def num_inliers(H, matched_points):
    n = 0
    for i in range(len(matched_points)):
        (kp_i, kp_j) = matched_points[i]
        
        p = np.array(kp_j.pt)
        d = np.linalg.norm(p - np.array(transform(H, kp_i.pt)))
        
        if d < 35: 
            n = n + 1
    return n


#ransac
def RANSAC(matched_points):
    max_inliers = 0
    H_opt = None
    
    for i in range(2000):
        #need at least 4 correspondences(matches) to compute it
        match = random.sample(matched_points, 4)
        #calculate homography
        H = homography(match)
        
        #get number of inliners
        n = num_inliers(H, matched_points)
        
        #set max_num inliners
        if n > max_inliers:
            max_inliers = n
            H_opt = H
    return H_opt, max_inliers
