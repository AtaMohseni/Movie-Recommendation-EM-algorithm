# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:55:49 2023

@author: ATA
"""

class Movie_Recommend():
    
    def __init__(self,movie_list,number_of_clusters):
        
        #list of movies for rating and recommendation
        self.movie_list = movie_list
        # number of moviegoer groups to project user into these groups 
        self.number_of_clusters = int(number_of_clusters)
        # initial probability of user belongs to moviegoer groups
        self.probZ = [1/self.number_of_clusters for n in range(self.number_of_clusters)]
        #probability of user likes a movie knowing that she/he belongs to 
        #any of moviegoer groups
        import numpy as np
        self.probRgivenZ = np.random.random_sample((len(self.movie_list),self.number_of_clusters))
        #list that rank the movies for user from most recommended movie to least recommended
        self.rank = []
        
        