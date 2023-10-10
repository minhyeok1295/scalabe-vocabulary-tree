# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:23:16 2022

@author: MinHyeok
"""
import os
import cv2
from helper import *
from sklearn.cluster import KMeans
import numpy as np
import pickle

class Node:
    def __init__(self):
        self.kmeans = None
        self.children = [] # number of children is branching factor, k
        self.num_des = {} # in the leaf node, save the number of descriptors of each img.
        self.index = 0 #node index 0 from the leftmost node and k**L -1 on the rightmost
        
        
class VTree:
    def __init__(self, k, L):
        self.des_counts = {} #records the number of descriptor vectors of image in each node
        self.num_img = 0 #number of imgs in the database
        self.all_img = [] #list of all imgs
        self.des_pair = [] #record of one descriptors and label with its image name
        self.tree = None
        self.num_leaf = 0 #also used to count index of node
        self.weights = None
        self.k = k 
        self.L = L
        self.database_vectors = {}

    #load the img from the database "./DVDcovers"
    def load_imgs(self, path):
        for filename in os.listdir(path):
            if "jpg" in filename:
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                kp, des = extract_features(img)
                self.num_img += 1
                self.all_img.append(filename)
                self.des_counts[filename] = np.zeros(self.k**self.L)
                
                for descriptor in des.tolist():
                    self.des_pair.append((descriptor, filename))
        self.weights = np.zeros(self.k**self.L) #number of leaf nodes
        
    def build_tree(self, k, L, pairs):
        all_descriptors = [tup[0] for tup in pairs]        
        node = Node()
        node.kmeans = KMeans(n_clusters = k, random_state=0)
        node.kmeans.fit(all_descriptors)
    
        if L == 0: # leaf node now count the number of descriptors of img and save
            node.index = self.num_leaf #node index, i..
            for tup in pairs: 
                img = tup[1]
                if img not in node.num_des:
                    node.num_des[img] = 1
                else:
                    node.num_des[img] += 1 
                    
            # increment the number of descriptor vector (m_i) of each image count dictionary
            for img, count in node.num_des.items():
                self.des_counts[img][node.index] += count

            #for each node save weights w_i = ln(N/N_i)
            self.weights[node.index] = np.log(self.num_img/len(node.num_des)) 
            self.num_leaf += 1
            return node
        else: #not leaf node so run the KMean again until it reaches leaf
            for i in range(k): #divide by branching factor k
                p = np.array(pairs, dtype=object)
                labels = np.array(node.kmeans.labels_)
                cluster = p[labels == i]
                child = self.build_tree(k, L-1, cluster)
                node.children.append(child)
        return node
    
    #save all distance vector, d_i, for each image for each node
    def compute_distance_vectors(self):
        for img in self.all_img:
            d = (self.des_counts[img] / np.sum(self.des_counts[img])) * self.weights 
            self.database_vectors[img] = d

            
            #d = (self.des_counts[img]) * self.weights
            #self.database_vectors[img] = d / np.sum(d)
    '''
    traverse down the tree with descriptor calculating the euclidean distance between
    KMeans center and descriptor. Follow down the node which has minimum distance.
    '''
    def get_leaf_node(self, root, des):
        if len(root.children) == 0:
            return root
        dist = np.linalg.norm(root.kmeans.cluster_centers_ - des, axis =1) #eud distance 
        idx = np.argmin(dist) #closest distance
        return self.get_leaf_node(root.children[idx], des)

    #compute query vector q_i.
    def compute_query_vectors(self, query_img):
        img = cv2.imread(query_img)
        kp, des = extract_features(img)
        
        n = np.zeros(self.num_leaf, dtype=int)
        target_imgs = []
        for d in des:
            node = self.get_leaf_node(self.tree, d)
            for img, count in node.num_des.items():
                if img not in target_imgs:
                    target_imgs.append(img)
            n[node.index] += 1
        #compute q_i = n_i * w_i
        q = np.zeros(self.num_leaf)
        for w in range(len(self.weights)):
            q[w] = n[w]  * self.weights[w] #n_i * w_i
        q = q / np.sum(q)
        return q, target_imgs

    def compute_score(self, query_vectors, target_imgs):
        score = np.zeros(len(target_imgs))
        for idx in range(len(target_imgs)):
            img = target_imgs[idx]
            d_i = self.database_vectors[img] 
            #since we used L1 normalization for each vector we use eq.5 from the paper.
            score[idx] = 2 + np.sum(np.abs(query_vectors - d_i) - np.abs(query_vectors) - np.abs(d_i))
        return score
    
    def save(self, database):
        file = open(database,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        
    def load(self, database):
        file = open(database,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
    
'''
compare query_img with top 10 images using the RANSAC.
All ransac related functions are from A4 panorama.
'''
def run_ransac(query_img, top_10_img):
    q_img = cv2.imread(query_img)
    best_img = None
    max_inlier = 0
    max_homography = None
    for database_img in top_10_img:
        
        d_img = cv2.imread("./DVDcovers" + "/" + database_img)
        matched_points = extract_and_match(d_img, q_img)
        H, num_inliers = RANSAC(matched_points)
        
        if num_inliers > max_inlier:
            max_inlier = num_inliers
            best_img = database_img
            max_homography = H
    print("matched img: {} - num inliers: {}".format(best_img, max_inlier))
    return best_img, max_homography
        
 
def visualize_homography(database_img_path, query_img_path, H, test_num):
    database_img = cv2.imread(database_img_path)
    query_img = cv2.imread(query_img_path)
    h,w = database_img.shape[:2]

    # define the reference points
    ref_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # affine transformation to display it on query_img
    dst = cv2.perspectiveTransform(ref_pts,H)
    result = cv2.polylines(query_img, [np.int32(dst)], True, (0,255,255), 2, cv2.LINE_AA)
    cv2.imwrite("{}_result_test.png".format(test_num), result)
    cv2.imwrite("{}_matched_cover.png".format(test_num), database_img)


if __name__ == "__main__":
    
    dir_path = "./DVDcovers"
    
    #List of test imgs
    #test_imgs = ["./test/image_01.jpeg","./test/image_02.jpeg","./test/image_03.jpeg","./test/image_04.jpeg","./test/image_05.jpeg","./test/image_06.jpeg","./test/image_07.jpeg"]
    #To test one img
    test_imgs = ["./test/image_07.jpeg"]
    
    #Initialize database
    
    vt = VTree(6,4) #change this value to (5,5) if you want to use database55.txt

    #uncomment this if you want to create new database with different k and L
    '''
    #load imgs and build vocabulary tree using k, and L given
    vt.load_imgs(dir_path)
    vt.tree = vt.build_tree(vt.k, vt.L, vt.des_pair)
    #save database
    vt.save("database{}{}.txt".format(vt.k, vt.L)
    
    '''
    #if database already, load database
    print("loading database")
    '''
    load database with k = 5, L = 5
    '''
    vt.load("database55.txt")
    '''
    load database with k = 6, L = 4
    '''
    #vt.load("database64.txt")

    #compute "d_i" for each node and each img.
    vt.compute_distance_vectors()

    j = 1
    for test_img in test_imgs:
        #compute q_i, query vector and retrieve all target img that was in the leaf node.
        query_vector, target = vt.compute_query_vectors(test_img)

        #compute the L1 normalization score given from the paper
        score = vt.compute_score(query_vector, target)
        
        #database images with lowest 10 score
        top_10_idx = np.argsort(score)[:10]
        top_10_img = [target[i] for i in top_10_idx]

        print("======================================================")
        print("top 10 images")
        for i in range(len(top_10_img)):
            print("{} - image: {}, score: {}".format(i, top_10_img[i], score[top_10_idx[i]]))
        print(" ")
        #run ransac
        print("running RANSAC")
        best_img, homography = run_ransac(test_img, top_10_img)
        
        #visualize homography on the test img
        visualize_homography(dir_path + "/" + best_img, test_img, homography, j)
        print("test result saved")
        print("======================================================")
        j += 1
 
