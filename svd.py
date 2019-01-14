import random

import numpy as np
from loguru import logger

from load_data import load_all_folds

n_latent_factors = 20
learning_rate = 0.01
regularizer = 0.03
max_epochs = 30
stop_threshold = 0.005


def get_triples(from_set, user_means, movie_means, mean_rating):
    triples = list()

    for user, movie_ratings in from_set.items():
        for movie, rating in movie_ratings.items():
            # Subtract the user's mean rating, movie's mean rating and add overall mean rating
            triples.append((user, movie, rating - user_means[user] - movie_means[movie] + mean_rating))

    return triples


def get_movies(from_set):
    movies = set()

    for user, movie_ratings in from_set.items():
        movies.update(movie_ratings.keys())

    return movies


def get_singular_vectors(n_movies, n_users):
    """ Return the left and right singular vectors """
    # Initialize singular value vectors
    A = np.random.rand(n_movies, n_latent_factors)
    B = np.random.rand(n_latent_factors, n_users)

    return A, B
    # Matrix multiplication results in a rating matrix
    # Should be size (movies, users)
    # m = np.matmul(A, B)


def calculate_rmse(on_set, movie_values, user_values, user_means, movie_means, mean_rating):
    # Compute the actual matrix R
    R = np.matmul(movie_values, user_values)

    n_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_instances += 0

        for movie, actual_rating in movie_rating.items():
            # Skip movies which we do not have means for
            # Not sure why, but the test set h
            if movie not in movie_means:
                continue

            # Get predicted rating
            # We add back the structure we removed earlier (user means and movie means)
            n_instances += 1
            predicted_rating = R[movie][user] + user_means[user] + movie_means[movie] - mean_rating
            residual = predicted_rating - actual_rating

            sum_squared_errors += pow(residual, 2)

    return np.sqrt(sum_squared_errors / n_instances)


def get_user_means(from_set):
    user_means = dict()

    for user, movie_ratings in from_set.items():
        user_means[user] = np.mean(list(movie_ratings.values()))

    return user_means


def get_movie_means(from_set):
    movie_rating_sum = dict()
    movie_n_ratings = dict()

    for user, movie_ratings in from_set.items():
        for movie, rating in movie_ratings.items():
            movie_n_ratings[movie] = movie_n_ratings.get(movie, 0) + 1
            movie_rating_sum[movie] = movie_rating_sum.get(movie, 0) + rating

    return {movie: r_sum / movie_n_ratings[movie] for movie, r_sum in movie_rating_sum.items()}


def get_average_rating(from_set):
    rating_sum = 0
    n_ratings = 0

    for user, movie_ratings in from_set.items():
        rating_sum += sum(rating for rating in list(movie_ratings.values()))
        n_ratings += len(movie_ratings)

    return rating_sum / n_ratings


def run(train):
    logger.info(f'Running with latent factors: {n_latent_factors}')

    # Construct the singular vectors
    movies = get_movies(train)
    movie_values, user_values = get_singular_vectors(max(movies) + 1, max(train.keys()) + 1)

    # Get mean ratings for movies and users, used for pre-processing
    # For every rating, we subtract the average of its user's ratings and average of its movie's rating
    # However, we add the mean rating in the entire system
    # This should give our optimisation a head start by removing some obvious structure
    user_means = get_user_means(train)
    movie_means = get_movie_means(train)
    mean_rating = get_average_rating(train)

    # Training instances are represented as a list of triples
    triples = get_triples(train, user_means, movie_means, mean_rating)
    last_rmse = None

    for epoch in range(max_epochs):
        # At the start of every epoch, we shuffle the dataset
        # Shuffling may not be strictly necessary, but is an attempt to avoid overfitting
        random.shuffle(triples)

        # Calculate RMSE for training set
        # Stop if change is below threshold
        rmse = calculate_rmse(train, movie_values, user_values, user_means, movie_means, mean_rating)
        if last_rmse and abs(rmse - last_rmse) < stop_threshold:
            break
        last_rmse = rmse

        logger.info(f'Epoch {epoch}, RMSE: {rmse}')

        for user, movie, rating in triples:
            error = rating - sum(movie_values[movie][i] * user_values[i][user] for i in range(n_latent_factors))

            # Update values in vector movie_values
            for k in range(n_latent_factors):
                gradient = error * user_values[k][user]
                # Update the movie's kth factor with respect to the gradient and learning rate
                # Large movie values are penalized using regularization
                movie_values[movie][k] += learning_rate * (gradient - regularizer * movie_values[movie][k])

            # Update values in vector user_values
            for k in range(n_latent_factors):
                gradient = error * movie_values[movie][k]
                # Update the user's kth factor with respect to the gradient and learning rate
                # Large user values are penalized using regularization
                user_values[k][user] += learning_rate * (gradient - regularizer * user_values[k][user])

    return movie_values, user_values, user_means, movie_means, mean_rating


def test_latent_factors(factors, train_folds, test_folds, n_folds=5):
    n_folds = min(n_folds, len(train_folds), len(test_folds))

    for factor in factors:
        global n_latent_factors
        n_latent_factors = factor
        rmse_results = []

        # Test for each fold
        for i in range(n_folds):
            train = train_folds[i]
            test = test_folds[i]

            movie_values, user_values, user_means, movie_means, mean_rating = run(train)
            rmse_results.append(calculate_rmse(test, movie_values, user_values, user_means, movie_means, mean_rating))

        logger.info(f'Finished test for {factor} latent factors, test RMSE values: {rmse_results}')
        logger.info(f'Average test RMSE: {np.mean(rmse_results)}')


if __name__ == "__main__":
    test_latent_factors([10, 15, 20, 25, 30, 40, 45, 50], *load_all_folds(), n_folds=2)
    # run(*load_fold(1))
