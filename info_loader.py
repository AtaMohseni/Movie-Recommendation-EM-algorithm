# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:24:25 2023

@author: ATA
"""

def MovieList():
    try:
        fhand = open("./hw8_movieTitles_fa18.txt")
    except:
        print ('the file does not exist')
        return None
    movie_list = []
    for line in fhand:
        movie_list.append(line.strip())
    return movie_list
    

def Movie_ratings():
    try:
        fhand = open('./hw8_ratings_fa18.txt')
        
    except:
        print ('one or more the files do not exist')
        return None
    ratings = []
    for line in fhand:
        ratings.append(list (line.split()))
        
    return ratings

def probRgivenZ():
    try:
        fhand = open('./hw8_probRgivenZ_init.txt')
        
    except:
        print ('one or more the files do not exist')
        return None
    probs = []
    for line in fhand:
        probs.append(list(map(float,list (line.split()) )))
        
    return probs
    
        