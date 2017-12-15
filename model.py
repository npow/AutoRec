import numpy as np
import tensorflow as tf
import tensorlayer as tl

class AutoRec:
    def __init__(self,
                 data,
                 batch_size=256,
                 hidden_size=500,
                 lr=0.001,
                 penalty=0.001,
                ):
        tf.reset_default_graph()
        tl.layers.clear_layers_name()
        num_movies = data['train']['ratings'].shape[1]
        
        self.sess = tf.Session()        
        self.data = data
        self.batch_size = batch_size
        
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='r')
        self.r_mask = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='r_mask')          
        
        l_in = tl.layers.InputLayer(self.r, name='input')
        l_encoder = tl.layers.DenseLayer(l_in,
                                         n_units=hidden_size,
                                         name='encoder',
                                         act=tf.nn.sigmoid,
                                         W_init=tf.truncated_normal_initializer(mean=0, stddev=0.05)
                                        )
        l_decoder = tl.layers.DenseLayer(l_encoder,
                                         n_units=num_movies,
                                         name='decoder',
                                         act=tl.activation.identity,
                                         W_init=tf.truncated_normal_initializer(mean=0, stddev=0.05)
                                        )
        network = l_decoder
        
        self.r_pred = l_decoder.outputs
        W_encoder = tl.layers.get_variables_with_name('encoder/W:0')[0]
        W_decoder = tl.layers.get_variables_with_name('decoder/W:0')[0]
        
        cost_reconstruction = tf.reduce_sum(tf.multiply((self.r - self.r_pred), self.r_mask) ** 2)
        cost_penalty = tf.reduce_sum(W_encoder ** 2) + tf.reduce_sum(W_decoder ** 2)
        
        self.cost = cost_reconstruction + penalty * 0.5 * cost_penalty
        self.train_op = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(self.cost, var_list=network.all_params)
            
        tl.layers.initialize_global_variables(self.sess)        
            
    def get_batch(self, dataset, index, n_batches):
        if index == n_batches - 1:
            r = dataset[self.batch_size*index:]
        else:
            r = dataset[self.batch_size*index:self.batch_size*(index+1)]
        r_mask = (r != 0).astype(np.float32)
        return r, r_mask
        
    def train(self, n_epochs=100, shuffle_batch=False):
        n_train_batches = (len(self.data['train']['ratings']) // self.batch_size) + 1
        epoch = 0
        while epoch < n_epochs:
            total_cost = 0
            for minibatch_index in xrange(n_train_batches):
                r, r_mask = self.get_batch(self.data['train']['ratings'], minibatch_index, n_train_batches)
                feed_dict = { self.r: r, self.r_mask: r_mask }
                cost, _ = self.sess.run([self.cost, self.train_op], feed_dict=feed_dict)                
                total_cost += cost
            print('epoch: %s, total_cost: %s' % (epoch, total_cost))
            for k in ['train', 'val', 'test']:
                self.test_model(k)
            epoch += 1
                
    def set_default_ratings(self, dataset, r_pred, r_mask):
        unseen_users = dataset['users'] - self.data['train']['users']
        unseen_movies = dataset['movies'] - self.data['train']['movies']
        for user in unseen_users:
            for movie in unseen_movies:
                if r_mask[user, movie] == 1:
                    r_pred[user][movie] = 3

    def test_model(self, k):
        dataset = self.data[k]
        r_mask = (dataset['ratings'] != 0).astype(np.float32)
        feed_dict = {
            self.r: dataset['ratings'],
            self.r_mask: r_mask
        }
        r_pred = self.sess.run([self.r_pred], feed_dict=feed_dict)
        self.set_default_ratings(dataset, r_pred, r_mask)
        rmse = np.sqrt(np.sum(np.multiply(r_pred - dataset['ratings'], r_mask) ** 2) / np.count_nonzero(r_mask))
        print('%s RMSE: %f' % (k.upper(), rmse))
