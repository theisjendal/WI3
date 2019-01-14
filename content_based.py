import json
import re
from math import log10, sqrt

from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords
from stemming.porter2 import stem


class TermDictionary:
    _stop_words = set(stopwords.words('english'))

    def __init__(self, product_reviews):
        self._internal_dict = {}
        self._vector_lengths = {}
        # Preprocess each product's merged review
        self._product_reviews = {product: self._preprocess(review) for product, review in product_reviews.items()}
        self._idfs = {}
        self._top_terms = list()

        logger.debug('Preprocessed reviews')

        # Incrementally construct index with each merged review
        for product, review in self._product_reviews.items():
            self._make_inverted_index(review, product)

        logger.debug('Created inverted matrix')

        # Pre-compute idf values
        n_reviews = len(self._product_reviews)
        for term in self._internal_dict:
            self._idfs[term] = log10(n_reviews / len(self._internal_dict[term]))

        logger.debug('Pre-computed idf values')

        # Consider only top-500 terms by term frequency
        term_frequencies = [(term, sum(list(self._internal_dict[term].values()))) for term in self._internal_dict]
        top_terms = sorted(term_frequencies, key=lambda tup: tup[1], reverse=True)
        self._top_terms = [tup[0] for tup in top_terms[:500]]

        # Pre-compute vector lengths for all products
        for product, review in self._product_reviews.items():
            squared_sum = sum(pow(self.get_tf_idf(term, product), 2) for term in self._top_terms)

            # The vector length is then the sqrt of the squared sum of components
            self._vector_lengths[product] = sqrt(squared_sum)

        logger.debug('Pre-computed vector lengths')

    def get_cosine_sim(self, a, b):
        # Compute the dot products of the vectors
        dot_product = sum(self.get_tf_idf(term, a) * self.get_tf_idf(term, b) for term in self._top_terms)

        return dot_product / (self._vector_lengths[a] * self._vector_lengths[b])

    def get_tf(self, term, identifier):
        return self._internal_dict[term].get(identifier, 0)

    def get_log_tf(self, term, identifier):
        tf = self.get_tf(term, identifier)
        if not tf:
            return 0

        return 1 + log10(tf)

    def get_idf(self, term):
        return self._idfs[term]

    def get_tf_idf(self, term, identifier):
        tf_log = self.get_log_tf(term, identifier)

        return tf_log * self.get_idf(term)

    def _make_inverted_index(self, document, identifier):
        """ Incremental approach to inverted index constructed """
        for term in document:
            if term not in self._internal_dict:
                self._internal_dict[term] = dict()

            self._internal_dict[term][identifier] = self._internal_dict[term].get(identifier, 0) + 1

    def _preprocess(self, text):
        # Convert document to lowercase and replace apostrophes
        # Apostrophes are removed because Treebank style tokenization splits them from their word
        # Periods are removed since Treebank does not remove them if they are between two words
        text = text.lower().replace('\'', '').replace('.', '')

        # Stemming using Porter's algorithm is also performed during preprocessing
        return [stem(token) for token in word_tokenize(text) if re.match(r'\w+', token) and token not in self._stop_words]


def load_reviews():
    return json.load(open('Musical_Instruments_5.json', 'r'))


def get_user_profile(product_reviews, reviewer_id):
    product_ratings = dict()

    for review in product_reviews:
        if review['reviewerID'] != reviewer_id:
            continue

        product_ratings[review['asin']] = review['overall']

    return product_ratings


def knn(user_profile, product_reviews, term_dict, k=3):
    if not user_profile:
        logger.error('Invalid user profile specified')

        return

    unseen_products = set(product_reviews).difference(set(user_profile))
    predicted_ratings = dict()

    for product in unseen_products:
        # Construct list of pairs from seen products to their similarity with the unseen product
        product_weights = [(seen, term_dict.get_cosine_sim(product, seen)) for seen in user_profile]

        # Sort the unseen product by its cosine similarity to the user's seen products
        # Take the top k products from this sorted list
        product_weights = sorted(product_weights, key=lambda tup: [1])[:k]

        # Predict the user's rating of the unseen item
        weighted_sum = sum(user_profile[tup[0]] * tup[1] for tup in product_weights)

        # To get the predicted rating, divide by the sum of absolute weights
        predicted_rating = weighted_sum / sum(abs(tup[1]) for tup in product_weights)

        # Save this product's predicted rating (to be sorted later)
        predicted_ratings[product] = predicted_rating
        #logger.info(f'Predicted rating of {product}: {predicted_rating}')

    # Return top-5 recommended items
    sorted_predictions = sorted(predicted_ratings.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return sorted_predictions


def run():
    reviews = load_reviews()

    # Combine reviews for each product
    product_reviews = dict()
    for review in reviews:
        product = review['asin']
        review = review['reviewText'].strip()

        product_reviews[product] = f'{review} {product_reviews.get(product, "")}'

    # Create term dictionary from merged product reviews
    term_dict = TermDictionary(product_reviews)
    logger.debug('Created term dictionary')

    # A user profile is generated as a dictionary from user's product reviews to the ratings
    user_profile = get_user_profile(reviews, 'A3DOXGBWDJ1MU0')

    # Finally, run the kNN algorithm on the user profile
    predictions = knn(user_profile, product_reviews, term_dict)

    # Print recommendations
    for product, predicted in predictions:
        logger.info(f'Predicted rating for {product}: {predicted}')


if __name__ == "__main__":
    run()
