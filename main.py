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
FLAGS = flags.FLAGS

def main(_):
    print(FLAGS.__flags)
    data = load_ratings('%s/ratings.dat' % FLAGS.input_dir)                
    args = {
        'batch_size': FLAGS.batch_size,
        'hidden_size': FLAGS.hidden_size,
        'lr': FLAGS.lr,
        'data': data,
        'penalty': FLAGS.penalty,
    }
    model = AutoRec(**args)
    model.train(n_epochs=FLAGS.n_epochs)

if __name__ == '__main__':
    tf.app.run()
