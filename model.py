import tensorflow as tf
import utility    as ut

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('driving_log_file', '', "csv file containing the driving log from the simulator")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('samples_per_epoch', 2000, "The number of sample by epoch.")
flags.DEFINE_integer('batch_size', 32, "The size of a batch.")

        
def main(_):

    # Extract info from our csv
    train_data = ut.extract_csv_data(FLAGS.driving_log_file)

    # Get the model
    model = ut.network_model()

    # Train it
    ut.train_model(  model, train_data
                   , FLAGS.batch_size
                   , FLAGS.epochs
                   , FLAGS.samples_per_epoch
                  )

    # Save the model
    ut.save_model(model)
        
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
