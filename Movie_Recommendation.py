# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:55:49 2023

@author: ATA
"""
import numpy as np
from info_loader import MovieList, Movie_ratings,probRgivenZ
class Movie_Recommend():
    """ class that represent naive Bayes Belief Network"""
    def __init__(self,movie_list,probRgivenZ,number_of_clusters):
        
        #list of movies for rating and recommendation (list of length n)
        self.movie_list = movie_list 
        # number of moviegoer groups to project user into these groups 
        self.number_of_clusters = int(number_of_clusters)
        # initial probability of user belongs to moviegoer groups, P(Z=i)
        self.probZ = [1/self.number_of_clusters for n in range(self.number_of_clusters)]
        
        #probability of user likes a movie knowing that she/he belongs to 
        #any of moviegoer groups which is P(Rj = 1 | Z = i ).
        #(you can initialize it randomly or import array of random probabilities)
        #Note that it has shape(n,z), n should match number of movies in
        # self.movie list, z should match self.number_of_clusters
        #self.probRgivenZ = np.random.random_sample((len(self.movie_list),self.number_of_clusters))
        self.probRgivenZ = np.array(probRgivenZ) 
        
        #normalized log-likelihood of movie rating data (will be 
        #calculated in training) 
        self.log_likelihood_normalized = None
                            
def product_probs_Rj_given_Zi(MVR, user_rating , z_cluster_index):
    """ Z cluster should be a number from 0 to (number_of_cluster - 1)
    this function calculate and return products of probabilities of user
    rating given type of movie goer
    
    -MVR: Movie_recommend instance
    -user rating: list of user ratings with length n, n is number of movies
    -z_cluster_index: integer from 0 to (number_of_cluster - 1) indicate
    particular movie-goer group"""
    
    product = 1
    for index, element in enumerate(user_rating):
        if element == "1":
            product *= MVR.probRgivenZ[index,z_cluster_index]
        elif element == "0":
            product *= (1 - MVR.probRgivenZ[index,z_cluster_index])
    return product
    
def E_step(MVR, data):
    """to compute and store the posterior probability for each user that
    he or she correspond to a particular type of movie-goer.will store these 
    probabilities in rhoit array.
    
    -MVR: Movie_recommend instance
    -data: list of lists for user ratings """
    
    #array of shape (z,t) where z is number of movie goer groups, t is number 
    #of user in data
    rhoit = np.zeros((MVR.number_of_clusters,len(data)))
    sum_log_likelihood_elements = 0
    
    for data_index, user_rating in enumerate(data):
        denominator = 0
        numerator_for_each_cluster = []
        for cluster_index, element in enumerate(MVR.probZ):
            products = product_probs_Rj_given_Zi(MVR, user_rating,cluster_index)
            denominator += element * products
            numerator_for_each_cluster.append(element*products)
            
        sum_log_likelihood_elements += np.log(denominator)
        rhoit[:,data_index] = (np.array(numerator_for_each_cluster)/denominator)
            
    return rhoit , sum_log_likelihood_elements

def Mstep(MVR,data):
    
    """ re-estimate the element in CPT of belief Network. i. P(Z=i) and 
    P(Rj = 1 | Z = i )
    
    -MVR: Movie_recommend instance
    -data: list of lists for user ratings""" 
    
    rhoit , sum_log_likelihood_elements= E_step(MVR,data)
    MVR.probZ = list(map(lambda n:n/len(data),list(np.sum(rhoit, axis = 1))))
    MVR.log_likelihood_normalized = sum_log_likelihood_elements/len(data)
    for movie_index,movie in enumerate(MVR.movie_list):
        for cluster in range(MVR.number_of_clusters):
            node_with_parents_numenator = 0    
            for user_index, user_rating in enumerate(data):
                if user_rating[movie_index] == '1':
                    node_with_parents_numenator += rhoit[cluster,user_index]
                elif user_rating[movie_index] == '?':
                    node_with_parents_numenator += rhoit[cluster,user_index]*MVR.probRgivenZ[movie_index,cluster]
            MVR.probRgivenZ[movie_index,cluster] = node_with_parents_numenator/(MVR.probZ[cluster]*len(data))
            
def train(MVR,data):
    """ function to check if users rating data is in list format. if yes
    then it starts updating CPT entries of Belief Network
    
    -MVR: Movie_recommend instance
    data: list of lists for user ratings"""
    
    if type(data) == list:
        Mstep(MVR,data)
    else:
        print('movie-rating data should be list format of length t users, each t element should be a list of n ratings corresponding to n title of movie')
        return None
     
if __name__ == "__main__":
    # Load required data to initialize Movie_Recommend class
    movie_list = MovieList()
    initial_probRgivenZ = probRgivenZ()
    rating_data = Movie_ratings()
    #initialize movie recommendation class
    MR  = Movie_Recommend(movie_list,initial_probRgivenZ, 5)
    
    #train several iteration and print normalize log-likelihood at some iteration
    printing = [0,1,2,4,8,16,32,64,128]
    for iteration in range(129):
        train(MR,rating_data)
        if iteration in printing:
            print(iteration,'  ',MR.log_likelihood_normalized)