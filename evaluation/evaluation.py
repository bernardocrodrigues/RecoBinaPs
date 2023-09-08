import statistics
from collections import defaultdict


def get_global_precision(predictions, relevance_threshold=1):
    """
    Returns the global precision, or micro-averaged precision, from a predictions list.

    Precision is defined as the fraction of relevant instances among the retrieved instances. In recommender systems,
    it measures the fraction of items that are liked by the user among the items that are recommended by the system.

    Precision = True Positives / (True Positives + False Positives)

    For example, if you have a recommender system that suggests movies to a user, and you have 100 movies in total,
    10 of which are liked by the user. If your system recommends 8 movies to the user, 4 of which are liked by the user
    (true positives), but also 4 of which are disliked by the user (false positives), then your precision is 4 / (4 + 4)
    = 0.5. This means that 50% of the movies that your system recommended were actually liked by the user.

    Global precision gives equal weight to each item, regardless of which user rated or was recommended it.
    """

    def is_relevant(measure):
        return measure >= relevance_threshold

    true_positives = 0
    false_positives = 0

    for _, _, true_rating, estimate, _ in predictions:
        if is_relevant(estimate):
            if is_relevant(true_rating):
                true_positives += 1
            else:
                false_positives += 1

    try:
        return true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        return 0


def get_user_averaged_precision(predictions, relevance_threshold=1):
    def is_relevant(measure):
        return measure >= relevance_threshold

    precisions = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        true_positives = 0
        false_positives = 0

        for estimate, true_rating in user_ratings:
            if is_relevant(estimate):
                if is_relevant(true_rating):
                    true_positives += 1
                else:
                    false_positives += 1

        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            pass
        else:
            precisions.append(precision)

    return statistics.mean(precisions)


def get_precision_at_k(predictions, relevance_threshold=1, k=20):
    def is_relevant(measure):
        return measure >= relevance_threshold

    precisions = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        relevant_itens_in_the_top_k = 0

        user_ratings.sort(key=lambda x: x[0], reverse=True)

        for estimate, true_rating in user_ratings[:k]:
            if is_relevant(true_rating):
                relevant_itens_in_the_top_k += 1

        precisions.append(relevant_itens_in_the_top_k / k)

    return statistics.mean(precisions)


def get_global_recall(predictions, relevance_threshold=1):
    """
    Returns the recall from a predictions list.

    Recall is defined as the fraction of relevant instances that were retrieved. In recommender systems,
    it measures the fraction of items that are liked by the user among all the items that are available.

    Recall = True Positives / (True Positives + False Negatives)

    For example, if you have a recommender system that suggests movies to a user, and you have 100 movies in total,
    10 of which are liked by the user. If your system misses 6 movies that are liked by the user (false negatives), then
    your recall is 4 / (4 + 6) = 0.4. This means that 40% of the movies that are actually liked by the user were
    recommended by your system.

    """

    def is_relevant(measure):
        return measure >= relevance_threshold

    true_positives = 0
    false_negatives = 0

    for _, _, true_rating, estimate, _ in predictions:
        if is_relevant(estimate):
            if is_relevant(true_rating):
                true_positives += 1
        else:
            if is_relevant(true_rating):
                false_negatives += 1

    return true_positives / (true_positives + false_negatives)


def get_user_averaged_recall(predictions, relevance_threshold=1):
    def is_relevant(measure):
        return measure >= relevance_threshold

    recalls = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        true_positives = 0
        false_negatives = 0

        for estimate, true_rating in user_ratings:
            if is_relevant(estimate):
                if is_relevant(true_rating):
                    true_positives += 1
            else:
                if is_relevant(true_rating):
                    false_negatives += 1
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            pass
        else:
            recalls.append(recall)

    return statistics.mean(recalls)


def get_recall_at_k(predictions, relevance_threshold=1, k=20):
    def is_relevant(measure):
        return measure >= relevance_threshold

    recalls = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        relevant_itens_in_the_top_k = 0
        total_relevant_itens = 0

        user_ratings.sort(key=lambda x: x[0], reverse=True)

        for estimate, true_rating in user_ratings[:k]:
            if is_relevant(true_rating):
                relevant_itens_in_the_top_k += 1

        for estimate, true_rating in user_ratings:
            if is_relevant(true_rating):
                total_relevant_itens += 1
        try:
            recalls.append(relevant_itens_in_the_top_k / total_relevant_itens)
        except ZeroDivisionError:
            pass

    return statistics.mean(recalls)
