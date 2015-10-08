# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 08:09:14 2015

@author: xule
"""

import SimpleCV as scv
import os
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import itertools
import cv2

front_caspath = "haarcascade_frontalface_default.xml"
profile_caspath = "haarcascade_profileface.xml"
faceFrontCascade = cv2.CascadeClassifier(front_caspath)
faceProfileCascade = cv2.CascadeClassifier(profile_caspath)

NO_TWO_MODEL = False

class Model:
    def __init__(self, img_dir):
        self._imgdir = img_dir
        self._extractors = self.__get_extractors()
        self._normalizer = Normalizer()
        self._face_normalizer = Normalizer()
        self._estimator = NearestNeighbors(n_neighbors=3)
        self._face_estimator = NearestNeighbors(n_neighbors=3)
        self._imgnames = []
        self._face_imgnames = []

    def __get_extractors(self):
        return [scv.HueHistogramFeatureExtractor(10),
                scv.EdgeHistogramFeatureExtractor(10),
                scv.HaarLikeFeatureExtractor(fname='haar.txt')]
                
    def __extract(self, imagepath):
        features = map(lambda e: e.extract(scv.Image(imagepath)), self._extractors)    
        return list(itertools.chain(*features))
        
    def normalize(self, fv):
        return self._normalizer.transform(fv)

    def match(self, imagepath):
        distances, indices = self._estimator.kneighbors(
            self.normalize(self.__extract(imagepath)))
        similarity = 1 - distances
        print 'Probability of liking (Model 1): %f' %(similarity.sum()/len(similarity[0]))
        print 'Top matching images:'
        for (sim, index) in zip(similarity[0], indices[0]):        
            print '\t%f %s' %(sim, self._imgnames[index])
        if NO_TWO_MODEL:
            return
        ##
        image = cv2.imread(imagepath)
        front_faces, profile_faces = self.__extractFaces(image)
        face_match = []
        for face_coords in front_faces:
            fv = self.__getfacefv(face_coords, image)
            fv = list(itertools.chain(*fv))
            try:
                fv = self._face_normalizer.transform(np.array(fv))
                distances, indexes = self._face_estimator.kneighbors(fv)
                distances = 1 - distances
            except:
                continue
            if len(distances) > 0:
                for d, i in zip(distances[0], indexes[0]):
                    face_match.append([d, self._face_imgnames[i]])
        for face_coords in profile_faces:
            fv = self.__getfacefv(face_coords, image)
            fv = list(itertools.chain(*fv))
            try:
                fv = self._face_normalizer.transform(np.array(fv))
                distances, indexes = self._face_estimator.kneighbors(fv)
                distances = 1 - distances
            except:
                continue
            if len(distances) > 0:
                for d, i in zip(distances[0], indexes[0]):
                    face_match.append([d, self._face_imgnames[i]])
        if len(face_match) > 0:
            p = sum(map(lambda x : x[0], face_match)) / len(face_match)
            print 'Probability of liking (Model 2): %s' %p
            for s, f in reversed(sorted(face_match, key=lambda x : x[0])):
                print '\t%f %s' %(s, f)
        
    def _getimages(self, dirpath):
        isvalid = lambda x : x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.bmp')
        for root, dirnames, filenames in os.walk(dirpath):
            for filename in filenames:
                fpath = os.path.join(root, filename)
                if not os.path.isfile(fpath):
                    continue
                if not isvalid(fpath):
                    continue
                self._imgnames.append(fpath)
        print 'Image count: %d' %len(self._imgnames)
        return self._imgnames
        
    def __extractFaces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [faceFrontCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                ), faceProfileCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                )]                
                
    def __getfacefv(self, face_coords, image):
        x, y, w, h = face_coords
        #print face_coords
        cv2.imwrite('/tmp/face.png', image[y:y+h, x:x+w])
        face = scv.Image('/tmp/face.png', cv2image = True)
        return map(lambda e: e.extract(face), self._extractors)

    def fit(self):
        ## build Model 1
        feature_vectors = map(lambda f: self.__extract(f),
                              self._getimages(self._imgdir))
        ## normalize the fv
        data = self._normalizer.fit_transform(np.array(feature_vectors))
        print 'Dataset size: {}'.format(data.shape)
        ##
        self._estimator.fit(data)
        if NO_TWO_MODEL:
            return
        ## build Model 2
        feature_vectors2 = []
        for imgpath in self._imgnames:
            image = cv2.imread(imgpath)
            front_faces, profile_faces = self.__extractFaces(image)
            for face_coords in front_faces:
                fv = self.__getfacefv(face_coords, image)
                feature_vectors2.append(list(itertools.chain(*fv)))
                self._face_imgnames.append(imgpath)
            for face_coords in profile_faces:
                fv = self.__getfacefv(face_coords, image)
                feature_vectors2.append(list(itertools.chain(*fv)))
                self._face_imgnames.append(imgpath)
        data2 = np.array(feature_vectors2)
        nans = np.isnan(data2).any(axis=1)
        fnames = np.array(self._face_imgnames)
        data2 = data2[~nans]
        fnames = fnames[~nans]
        self._face_imgnames = list(fnames)
        data2 = self._face_normalizer.fit_transform(data2)
        print 'Face dataset size: {}'.format(data2.shape)
        self._face_estimator.fit(data2)
        
if __name__ == '__main__':
    build_model = False
    if build_model:
        model = Model('/media/lvm/xule/pictures/gallery')
        model.fit()
        output = open('gallery_2model.pkl', 'wb')
        pickle.dump(model, output)
        output.close()
    else:
        pkl_file = open('gallery_2model.pkl', 'rb')
        model = pickle.load(pkl_file)
        pkl_file.close()
        
    test_image_name = '/home/xule/Downloads/9EGX5.jpg'
    print 'Matching: %s' %test_image_name
    model.match(test_image_name)
