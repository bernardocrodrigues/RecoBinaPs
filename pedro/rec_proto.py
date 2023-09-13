import numpy as np
import h5py
from sklearn.model_selection import KFold
from rec_functions import item_similarity, item_target_similarity, user_pattern_similarity, get_metrics
import timeit

def main(t_spar = 5.0, n_splits = 5, n_top_pat = 5, k_top_items = 5, N_eval = 25, t_rat = 3.5, run=0):

    # Input Parameters 

    # t_spar : Threshold for sparsity on pattern set selection
    # n_splits : K-fold parameter
    # n_top_pat : Number of most similar pattern sets to be selected
    # k_top_items : k for IBKNN approach
    # N_eval : Parameter for @N metrics
    # t_rat : Threshold for rating recommendation
    
    seed = 1
    np.random.seed(seed)

    dataset = 'latest-small'

    # Read ratings matrix

    filename = 'data/ml-'+ dataset +'.h5'
    with h5py.File(filename, 'r') as f:

        ratings_matrix = f['data'][()]
        
    n_rows, n_cols = ratings_matrix.shape

    # Read pattern sets 

    file = 'data/ml-'+dataset+'_t'+str(int(t_rat*10))+'.binaps.patterns'
    patterns = []
    with open(file, 'r') as f:
        array = ''
        for line in f:
            if(line[-2] != ']'):
                array += line[1:-1]+' '
                continue
            else:
                array += line[1:-2]
                cols = np.fromstring(array, dtype=int, sep=' ')
                patterns.append(cols)
                array = ''
                
    # Calculate the data sparsity for each pattern set
    n_pat = len(patterns)
    sparsity = np.zeros(n_pat)
    for i in np.arange(n_pat):
        ratings = ratings_matrix[:, patterns[i]]
        sparsity[i] = 100*np.sum(ratings > 0)/ratings.size
                
    # Select pattern sets with sparsity above a chosen threshold
    ind = np.arange(n_pat)[sparsity > t_spar]
    sel_patterns = [patterns[i] for i in ind]
    n_sel_pat = len(sel_patterns)

    # Find coordinates of non-zero ratings
    Y, X = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    ids = ratings_matrix > 0
    X = X[ids] # Rows of non-zero ratings
    Y = Y[ids] # Columns of non-zero ratings

    n_ratings = len(X)
    kf = KFold(n_splits, shuffle=True)

    # Initializes results file
    with h5py.File('data/ml-'+ dataset +'_results'+str(run)+'.h5', 'w') as f:
        f['t_spar'] = t_spar
        f['n_splits'] = n_splits
        f['n_top_pat'] = n_top_pat
        f['k_top_items'] = k_top_items
        f['N_eval'] = N_eval
        f['t_rat'] = t_rat
        
    # Initialize metrics

    MAE = np.zeros(n_splits)
    prec = np.zeros(n_splits)
    recall = np.zeros(n_splits)
    ndcg = np.zeros(n_splits)
        
    start = timeit.default_timer()

    # K-fold training

    for fold, (train_index, test_index) in enumerate(kf.split(X)):

        #X_train = X[train_index]
        #Y_train = Y[train_index]
        
        X_test = X[test_index]
        Y_test = Y[test_index]
        n_test = len(test_index)
        
        data = np.copy(ratings_matrix)
        data[X_test, Y_test] = 0.0 # Erases ratings from test set
        predictions = np.copy(data) # Matrix with predictions

        sim_item = np.full((n_cols, n_cols), -np.inf) # Initializes item similarity matrix
        
        mean_rating = np.mean(data, axis=1, where = data>0.0) # Mean rating for each user

        # Calculate user-pattern similarity matrix
        sim_user = np.zeros((n_rows, n_sel_pat)) 
        for user in np.arange(n_rows):
            for pattern in np.arange(n_sel_pat):
                sim_user[user, pattern] = user_pattern_similarity(data, user, sel_patterns[pattern])

        for user, item in zip(X_test, Y_test):
            
            # Find the n_top_pat most similar pattern sets to the current user
            id_sort = np.argsort(sim_user[user])
            
            # If there is no similarity between the user and any pattern, no prediction is calculated
            if(sim_user[user, id_sort[-1]] == 0):
                pred = 0.0
            
            # Otherwise, proceed to calculate prediction
            else:
            
                top_pat = id_sort[-n_top_pat:] 

                # Merge the n_top_pat most similar pattern sets to the current user

                merged_pat = np.array([], dtype=int)
                for pat in top_pat:
                    merged_pat = np.append(merged_pat, sel_patterns[pat])
                merged_pat = np.unique(merged_pat)

                # Find the items from merged_pat rated by the target user 
                id_merge = data[user, merged_pat] > 0 
                rated_items = merged_pat[id_merge] 

                item_target_similarity(data, item, rated_items, sim_item) # Calculate item similarity between target item and rated items
                id_sort = np.argsort(sim_item[item, rated_items]) # Sort rated items by similarity with target item
                
                # If there is no similarity between the target item and any other item, no prediction is calculated
                if(sim_item[item, rated_items[id_sort[-1]]] == 0):
                    pred = 0.0
                
                # Otherwise, proceed to calculate prediction
                else:
                            
                    # Neighborhood: k most similar items to target item
                    id_neigh = id_sort[-k_top_items:] 
                    neigh = rated_items[id_neigh]

                    # Calculate predicitions based on the neighborhood
                    num = 0.0
                    den = 0.0
                    for j in neigh:
                        num += (data[user, j] - mean_rating[user]) * sim_item[item, j]
                        den += abs(sim_item[item, j])
                        
                    pred = mean_rating[user] + num / den

            predictions[user, item] = pred
            
        # Evaluate metrics
        
        metrics = get_metrics(predictions, ratings_matrix, X_test, Y_test, t_rat, N_eval)
        
        MAE[fold] = metrics['MAE']
        prec[fold] = metrics['prec']
        recall[fold] = metrics['recall']
        ndcg[fold] = metrics['ndcg']
        
        stop = timeit.default_timer()
        te = stop - start
        print(f'Saving predictions for execution {fold+1}, time elapsed: {int(te/60):02d}m{int(np.mod(te,60)):02d}s')
        # Save predictions to results file
        with h5py.File('data/ml-'+ dataset +'_results'+str(run)+'.h5', 'a') as f:
            #f['data'+str(fold)] = data
            f['train_index'+str(fold)] = train_index
            f['test_index'+str(fold)] = test_index
            f['predictions'+str(fold)] = predictions[X_test, Y_test]

    with h5py.File('data/ml-'+ dataset +'_results'+str(run)+'.h5', 'a') as f:
        f['MAE'] = MAE
        f['prec'] = prec
        f['recall'] = recall
        f['ndcg'] = ndcg
        
    print(f'Run {run}: MAE: {MAE}\nPrecision @{N_eval}: {prec}\nRecall @{N_eval}: {recall}\nNDCG @{N_eval}: {ndcg}')

if __name__ == "__main__":
    main()
