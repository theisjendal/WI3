from math import sqrt

import numpy as np
from loguru import logger

from content_based import load_reviews

reviews = load_reviews()

# Collect ratings from all users
ratings_matrix = dict()
for review in reviews:
    reviewer = review['reviewerID']
    product = review['asin']
    rating = review['overall']

    if reviewer not in ratings_matrix:
        ratings_matrix[reviewer] = dict()

    ratings_matrix[reviewer].update({product: rating})

# Compute average ratings
mean_ratings = dict()
for user, ratings in ratings_matrix.items():
    mean_ratings[user] = np.mean(list(ratings.values()))


def user_sim(a, b):
    mutual_purchases = set(ratings_matrix[a]).intersection(set(ratings_matrix[b]))

    # If they only have one mutual purchase, the mean adjusted rating will be 0
    # If they have no mutual purchases, we define their similarity to be 0
    if len(mutual_purchases) < 3:
        return 0

    # Calculate the sum of the users' ratings of products
    # Mean-adjusted cosine similarity (Pearson correlation)
    def mean_adjusted_rating(u, p):
        return ratings_matrix[u][p] - mean_ratings[u]

    # Compute the mean adjusted dot product of ratings
    ratings_dot_product = sum(mean_adjusted_rating(a, p) * mean_adjusted_rating(b, p) for p in mutual_purchases)

    a_vector = sqrt(sum(mean_adjusted_rating(a, p) ** 2 for p in mutual_purchases))
    b_vector = sqrt(sum(mean_adjusted_rating(b, p) ** 2 for p in mutual_purchases))

    pearson_correlation = ratings_dot_product / (a_vector * b_vector)

    return pearson_correlation


def prediction(for_user, on_product):
    # Only consider neighbors who have bought the product
    neighbors = set()
    for other_user, products in ratings_matrix.items():
        if for_user != other_user and on_product in products:
            # Consider only users with variance > 0 in their ratings
            rating_variance = np.var(list(products.values()))

            if rating_variance > 0:
                neighbors.add(other_user)

    # Compute similarity to all neighbors
    similarities = dict()

    for other_user in neighbors:
        if for_user == other_user:
            continue

        similarities[other_user] = user_sim(for_user, other_user)

    # Sum weights to be used when creating weights
    # We are weighting such that similar users' ratings are weighted higher
    sum_weights = np.sum(list(similarities.values()))

    def weight(other):
        return similarities[other] / sum_weights

    # Add the current user's mean rating to the prediction
    rating_pred = mean_ratings[for_user] + sum(weight(other) * (ratings_matrix[other][on_product] - mean_ratings[other]) for other in neighbors)

    return rating_pred


def print_recommended(for_user):
    # Predict ratings of items
    predicted_ratings = dict()

    # Get all unique products
    unique_products = set()
    for u, r in ratings_matrix.items():
        unique_products.update(set(r.keys()))

    # Only get predictions for products the user has not seen
    for item in unique_products:
        if item not in ratings_matrix[for_user]:
            predicted_ratings[item] = prediction(for_user, item)

    # Return top-5 recommended items
    sorted_predictions = sorted(predicted_ratings.items(), key=lambda kv: kv[1], reverse=True)[:5]

    for item, predicted in sorted_predictions:
        logger.info(f'Predicted rating for {item}: {predicted}')


if __name__ == "__main__":
    # Define user similarity
    users = list(ratings_matrix.keys())

    print_recommended('A3DOXGBWDJ1MU0')
