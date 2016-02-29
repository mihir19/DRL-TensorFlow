import tensorflow as tf
import numpy as np
import random

N = 10000
K = 3
def create_data():
    p = tf.Variable(tf.random_uniform([N,2]))
    return p

def Kmeans(vector, cluster):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        initial = random.sample(vector, cluster)
        print initial

if __name__=="__main__":
    points = create_data()
    Kmeans(points, K)
    model = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(model)
        print(session.run(points))
