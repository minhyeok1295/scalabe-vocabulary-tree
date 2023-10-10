# scalabe-vocabulary-tree

### Introduction:
This project is based on the paper by Nister, Stewnius, Scalable Recognition with a Vocabulary Tree. Given test image, it retrieves best matching DVD cover from the database and visualize its homgraphy
1. Built database Called VTree (Vocabulary Tree) which stores the descriptors of all images in "DVDcovers" folder. The tree has branching factor 'k', which is also used in K-mean level, L
2. After extracting features from the images in database, all descriptors for each DVD cover is used to build the tree by running K-menas with 'k' value given. The group of descriptors will be partitioned into 'k' group, then for each group, we run the K-mean and partition recursively until we reach the 'L' levels. The clusters in the leaf node act as word.
3. Query image vector (q_i) is the product of weight of node i and number of descriptors of query image. 
4. After extracting features of query image, and for each descriptor vector, we traverse down the vocabulary tree by calculating the euclidean distance between descriptor vector and tree node’s cluster centers (centroid). Once it reaches the leaf node, it records all images in the leaf node and also calculates the ‘q_i’ given weights and number of descriptor vectors that belongs to the query image
5. Database image vector (m_i) is the product of weight of node i and number of descriptor of database 
image. Both vectors are normalized by dividing each vector by sum of all vectors.
6. Using the scoring method defined in the paper, we take in the two descriptor (q_i, d_i) for query and 
database image. In the paper, David stated that L-1 normalization gives better results than L-2 so we 
used above equation, (5) to calculate the score between query images and database images. Then by 
using sorting method, retrieved 10 lowest score for the RANSAC method
