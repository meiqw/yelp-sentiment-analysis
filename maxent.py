# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import nltk
from nltk.corpus import stopwords
import numpy as np
import math
import scipy.misc
from random import shuffle, seed

class MaxEnt(Classifier):

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.001, 30)

    def features_labels(self, instances):
        self.labels = []
        self.dict = {}
        count_dict = {}

        for instance in instances:
            # append labels to labels list
            if instance.label not in self.labels:
                self.labels.append(instance.label)

            # count words in each instance's features

            if len(instance.features()) == 2:
                word = instance.features()[0] + instance.features()[1]
                if word not in self.dict:
                    self.dict[word] = 1
            else:
                for word in instance.features():
                    word = word.lower()
                    if word not in count_dict:
                        count_dict[word] = 1
                    else:
                        count_dict[word] += 1


        # store words in count_dict if the word appears more than 10 times
        # and the key is not in stopwords

        #stop_words = set(stopwords.words('english'))
        stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
        for key in count_dict:

            if count_dict[key] > 10 and key not in stop_words:
                self.dict[key] = count_dict[key]

        # add a bias at the end of the word counts dictionary
        self.dict["Bias"] = 1

        # print out labels and features
        if len(count_dict) > 1:
            print("features shown: ", len(count_dict))
        print("actual features size: ", len(self.dict))
        print("labels size: ", len(self.labels))

        # build a matrix out of label counts and feature counts
        self.theta = np.zeros((len(self.labels), len(self.dict)))

    # build a feature vector out of the feature counts dictionary
    def featurization(self, instance):
        feature_vector = np.zeros((1, len(self.dict)))

        if len(instance.features()) == 2:
            word = instance.features()[0] + instance.features()[1]
            if word in self.dict:
                feature_vector[0][list(self.dict).index(word)] = 1
        else:
            for word in instance.features():
                if word in self.dict:
                    feature_vector[0][list(self.dict).index(word)] = 1

        feature_vector[0][len(self.dict)-1] = 1
        return feature_vector[0]

    # build a feature matrix out of feature vector
    def feature_matrix(self, label, instance):
        f = np.zeros((len(self.labels),len(self.dict)))
        if len(instance.feature_vector) == 0:
            instance.feature_vector = self.featurization(instance)
        f[self.labels.index(label)] = instance.feature_vector

        return f

    def posterior(self, label, instance):
        unnormalized_score1 = np.dot(self.theta.flatten(), self.feature_matrix(label, instance).flatten())
        unnormalized_score2 = []
        for y in self.labels:
            unnormalized_score2.append(np.dot(self.theta.flatten(), self.feature_matrix(y,instance).flatten()))
        return math.exp(unnormalized_score1-scipy.misc.logsumexp(unnormalized_score2))

    def negative_loglikelihood(self, minibatch):
        loglikelihood = 0
        for instance in minibatch:
            loglikelihood += np.dot(self.theta.flatten(), self.feature_matrix(instance.label, instance).flatten())

            unnormalized_score2 = []
            for y in self.labels:
                unnormalized_score2.append(np.dot(self.theta.flatten(), self.feature_matrix(y, instance).flatten()))

            loglikelihood -= scipy.misc.logsumexp(unnormalized_score2)
        return -loglikelihood + self.regularization(0)

    def regularization(self, lamda):
        return (lamda / 2) * np.dot(self.theta.flatten(), self.theta.flatten())

    def chop_up(self, training_set, batch_size):
        minibatches = []
        for i in range(0, len(training_set) // batch_size):
            minibatches.append(training_set[i * batch_size:(i + 1) * batch_size])
        if len(training_set) % batch_size != 0:
            minibatches.append(training_set[len(training_set) // batch_size * batch_size:])
        return minibatches

    def compute_gradient(self, minibatch):
        gradient = np.zeros((len(self.labels), len(self.dict)))
        for instance in minibatch:
            gradient += self.feature_matrix(instance.label, instance)
            for y in self.labels:
                gradient -= (self.posterior(y, instance)*self.feature_matrix(y, instance))

        return gradient

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        self.features_labels(train_instances)
        print("batch_size: ", batch_size)

        iter = 0
        theta_list = [0]
        dev_loss_list = [0]

        x_array = []
        y_array = []
        accu_lst = []

        while(iter < 50):
            iter += 1
            minibatches = self.chop_up(train_instances, batch_size)

            for minibatch in minibatches:
                delta_theta = self.compute_gradient(minibatch)
                self.theta += (delta_theta * learning_rate)

                '''
                if iter == 25:
                     x += len(minibatch)
                     x_array.append(x)
                     correct_number = 0
                     for instance in dev_instances:
                         if self.classify(instance) == instance.label:
                            correct_number += 1
                     y_array.append(correct_number / len(dev_instances))
                '''

            theta_list.append(self.theta)
            dev_loss = self.negative_loglikelihood(dev_instances)
            dev_loss_list.append(dev_loss)
            acc_number = 0
            for instance in dev_instances:
                if self.classify(instance) == instance.label:
                    acc_number += 1

            print("iter", iter, ", train loss:", self.negative_loglikelihood(train_instances),", dev loss", dev_loss, "dev acc:", acc_number / len(dev_instances))
            accu_lst.append(acc_number / len(dev_instances))

            if iter > 4 and dev_loss_list[iter] > dev_loss_list[iter - 1] and dev_loss_list[iter - 1] > dev_loss_list[iter - 2]:
                self.theta = theta_list[iter - 2]
                break

            shuffle(train_instances)

        max_accu = max(accu_lst)
        print(max_accu)
        #print(x_array)
        #print(y_array)


    def classify(self, instance):
        p = 0
        for i in range(0, len(self.labels)):
            if self.posterior(self.labels[i], instance) > p:
                p = self.posterior(self.labels[i], instance)
                result = self.labels[i]
        return result

    
