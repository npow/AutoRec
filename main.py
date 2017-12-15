import tensorflow as tf
from model import AutoRec
from utils import load_ratings

flags = tf.app.flags
flags.DEFINE_string('input_dir', 'ml-1m', 'input directory containing movielens files')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('hidden_size', 500, 'hidden size')
flags.DEFINE_integer('n_epochs', 50, 'num epochs')
flags.DEFINE_float('lr', 0.005, 'learning rate')
flags.DEFINE_float('penalty', 1, 'regularization penalty')
flags.DEFINE_float('keep', 0.9, 'dropout keep probability')
flags.DEFINE_float('random_state', 1234, 'random state seed')
FLAGS = flags.FLAGS

def main(_):
    print(FLAGS.__flags)
    FLAGS.data = load_ratings('%s/ratings.dat' % FLAGS.input_dir, random_state=FLAGS.random_state)
    model = AutoRec(**FLAGS.__flags)
    model.train(n_epochs=FLAGS.n_epochs)

if __name__ == '__main__':
    tf.app.run()
