import numpy as np
from sklearn.metrics import ndcg_score

def item_similarity(data, mean_rating, i, j):
    
    ind = (data[:, i] * data[:, j] > 0) # index of users with ratings for both items
    if(~ind.any()): # if no users in common, zero similarity
        return 0.0
    ratings = data[ind][:, [i,j]] # matrix with common ratings for both users
    #ratings = data[:, [i,j]]
    means = mean_rating[ind] # mean ratings for these users
    
    # Calculating similarity between items i and j
    cent_rat = ratings.T - means
    num = np.sum(cent_rat[0]*cent_rat[1])
    sq_rat = cent_rat**2
    den = np.sqrt(np.prod(np.sum(sq_rat, axis=1)))
    
    if(den == 0):
        #print(f'Degenerado:  {i,j}')
        return 1.0
    
    sim = num/den

    return sim
    
def item_set_similarity(data, pattern, sim_item):
    # Calculate item similarity between pairs of items from a given pattern set
    # Inputs: data with ratings, lisf of items from pattern set and item similarity matrix
    # Item similarity matrix is initialized with inf's.
    
    mean_rating = np.mean(data, axis=1, where = data>0.0)
    n_items = len(pattern)
    
    for j in np.arange(n_items):
        for k in np.arange(j+1, n_items):
            if(np.isinf(sim_item[pattern[j], pattern[k]])): # If not inf, similarity of the pair was previously calculated
                sim = item_similarity(data, mean_rating, pattern[j], pattern[k])
                sim_item[pattern[j], pattern[k]] = sim
                sim_item[pattern[k], pattern[j]] = sim
                
def item_target_similarity(data, target, pattern, sim_item):
    # Calculate item similarity between target item and each item from a given pattern set
    # Inputs: data with ratings, target item, lisf of items from pattern set and item similarity matrix
    # Item similarity matrix is initialized with inf's.
    
    mean_rating = np.mean(data, axis=1, where = data>0.0)
    n_items = len(pattern)
    
    for i in np.arange(n_items):
        if(target != pattern[i] and np.isinf(sim_item[target, pattern[i]])): # If not inf, similarity of the pair was previously calculated
            sim = item_similarity(data, mean_rating, target, pattern[i])
            sim_item[target, pattern[i]] = sim
            sim_item[pattern[i], target] = sim
            
def user_pattern_similarity(matrix, user, pattern, thr=0.0):
    
    # Returns user similarity to a given pattern
    # Input: ratings matrix, user index, pattern set and rating threshold
    
    #num = np.count_nonzero(matrix[user, pattern])
    num = np.sum(matrix[user, pattern] > thr)
    den = pattern.size
    sim = num/den
    
    return sim
    
# Calculates metrics for generated predictions
def get_metrics(predictions, ratings_matrix, X_test, Y_test, t_rat, N_eval):

	# Input:
	# predictions: ratings matrix with predicted ratings
	# ratings_matrix: true ratings
	# X_test and Y_test: list of indexes of the matrix for the test set
	#                    (X_test[i], Y_test[i]) are the coordinates for the i-th predicted rating
	# t_rat: threshold for rating recommendation
	# N_eval: number of top recommendations to be considered for metrics
	
	# Output: 
	# Dictionary with the following metrics:
	# MAE: Mean Absolute Error between predictions and true ratings
	# prec: Precision at N for the top N_eval recommended items
	# recall: Recall at N for the top N_eval relevant items
	# ndcg: Normalized Discounted Cumulative Gain

    n_rows, n_cols = ratings_matrix.shape

    n_test = len(X_test)

    diff = ratings_matrix - predictions
    MAE = np.sum(abs(diff))/n_test
    
    # Initialize counters for precision and recall metrics
    n_rec = 0
    n_pos = 0
    n_rel = 0
    
    cum_ndcg = 0.0 # Cumulative normalized discounted cumulative gain
    
    i = 0
    
    for user in np.arange(n_rows):

        prev_i = i

        while(X_test[i] == user):
            i += 1
            if(i == n_test):
                break

        ind_user = Y_test[prev_i:i] # items predicted for user

        pred = predictions[user, ind_user] # predictions for user
        ind_rec = np.where(pred >= t_rat)[0] 
        pred = pred[ind_rec] # ratings of recommended items
        ind_rec = ind_user[ind_rec] # recommended items
        top_N = np.argsort(pred)[-N_eval:]
        top_rec = ind_rec[top_N] # top N_eval recommended items

        n_top = len(top_N)
        n_rec += n_top # number of top recommendations
        n_pos += np.sum(ratings_matrix[user, top_rec] >= t_rat) # number of relevant items among recommended
        
        if(n_top > 1):
            y_true = ratings_matrix[user, top_rec][np.newaxis,:]
            y_score = predictions[user, top_rec][np.newaxis,:]
            cum_ndcg += ndcg_score(y_true, y_score)

        rat = ratings_matrix[user, ind_user] # actual ratings for user
        ind_rel = np.where(rat >= t_rat)[0] 
        n_rel += len(ind_rel) # number of top relevant items
        
        if(i == n_test):
            break

    prec = 100*n_pos/n_rec
    recall = 100*n_pos/n_rel
    ndcg = cum_ndcg/n_rows
    
    metrics = {'MAE': MAE,
               'prec': prec,
               'recall': recall,
               'ndcg': ndcg}
               
    return metrics
            

