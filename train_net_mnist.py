import os
import argparse
import numpy as np
import datetime
import tensorflow as tf
from absl import logging

from models.models import ArcFaceModel

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_mode', type=str, choices=['eager_mode', 'fit_mode'], help='mode to train',
                        default='eager_mode')
    parser.add_argument('--max_epoch', type=int, help='epoch to train the network', default=10)
    parser.add_argument('--image_size', help='the image size', default=28)
    parser.add_argument('--image_channels', type=int, help='the image size', default=1)
    parser.add_argument('--backbone_model', type=str, choices=['ResNet50', 'MobileNetV2', 'Xception', 'vgg8'],
                        help='The backbone model', default='vgg8')  # MNIST dataset can only be used with 'vgg8'
    parser.add_argument('--loss_head_type', type=str, choices=['ArcHead', 'NormHead'], help='The loss type',
                        default='ArcHead')
    parser.add_argument('--class_number', type=int, help='class number depend on your training datasets', default=10)
    parser.add_argument('--embedding_size', type=int, help='Dimensionality of the embedding.', default=3)
    parser.add_argument('--initial_learning_rate', type=float, help='Initial learning rate', default=1e-1)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'SGD'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--reg_weight_decay', type=float, help='weight decay for regression loss', default=0.1)
    parser.add_argument('--train_batch_size', type=int, help='batch size to train network', default=128)
    parser.add_argument('--summary_path', help='the summary file save path', default='output')
    parser.add_argument('--save_path', help='the ckpt file save path', default='./output/ckpt/')

    args = parser.parse_args()
    return args


def test_step(batch_data):
    backbone_model = ArcFaceModel(size=args.image_size,
                                  backbone_type=args.backbone_model,
                                  num_classes=args.class_number,
                                  head_type=args.loss_head_type,
                                  channels=1,
                                  embd_shape=3,
                                  training=False)

    ckpt_path = tf.train.latest_checkpoint(
        os.path.join(args.save_path, args.backbone_model + '_' + args.loss_head_type))
    if ckpt_path is not None:
        print('[*] load ckpt from {}'.format(ckpt_path))
        backbone_model.load_weights(ckpt_path)
    else:
        print('[*] Cannot find ckpt.')
        exit()

    images, labels = batch_data

    embeddings = backbone_model.predict(images)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + tf.keras.backend.epsilon()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    for c in range(len(np.unique(test_labels))):
        ax1.plot(embeddings[test_labels == c, 0], embeddings[test_labels == c, 1],
                 embeddings[test_labels == c, 2], '.', alpha=0.1)
    plt.show()


if __name__ == '__main__':
    args = get_parser()

    # Setting GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    strategy = tf.distribute.MirroredStrategy()
    print('available GPUs: {}'.format(strategy.num_replicas_in_sync))

    # Setting Log
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    # Load Dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = (train_images[:, :, :, np.newaxis].astype('float32') - 127.5) / 127.5
    test_images = (test_images[:, :, :, np.newaxis].astype('float32') - 127.5) / 127.5

#    with strategy.scope():
    # Load Model
    model = ArcFaceModel(size=args.image_size,
                         backbone_type=args.backbone_model,
                         channels=1, num_classes=args.class_number,
                         head_type=args.loss_head_type,
                         embd_shape=args.embedding_size,
                         logist_scale=10,
                         training=True)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

    # Define optimizer & learning rate schedule
    starter_learning_rate = args.initial_learning_rate
    end_learning_rate = args.initial_learning_rate * 0.01
    decay_steps = args.max_epoch * len(train_images) / args.train_batch_size
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate,
                                                                     decay_steps,
                                                                     end_learning_rate,
                                                                     power=1)

    if args.optimizer == 'ADAGRAD':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_fn)
    elif args.optimizer == 'ADADELTA':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate_fn)
    elif args.optimizer == 'ADAM':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    elif args.optimizer == 'RMSPROP':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
    elif args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    else:
        raise ValueError('Invalid optimization algorithm')

    # Setting summary path
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.summary_path, 'tb_' + current_time)

    if args.train_mode == 'eager_mode':
        import math

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(len(train_images))
        train_dataset = train_dataset.batch(args.train_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)

        @tf.function
        def train_step(batch_data):
            input_batch, label_batch = batch_data

            with tf.GradientTape() as tape:
                logits = model(batch_data, training=True)
                reg_loss = tf.reduce_sum(model.losses) * args.reg_weight_decay
                pred_loss = loss_fn(label_batch, logits)
                total_loss = pred_loss + reg_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_accuracy.update_state(label_batch, logits)

            return total_loss

        # Start training
        count = 0
        for epoch in range(args.max_epoch):
            step = 0

            for batch_data in train_dataset:
                total_losses = train_step(batch_data)

                # Save tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/train_loss', total_losses, step=count)
                    tf.summary.scalar('loss/train_acc', train_accuracy.result(), step=count)

                print('\rEpoch: {}/{}, step: {}/{}, loss: {:.5f}, acc: {:f}'.format(
                    epoch + 1,
                    args.max_epoch,
                    step,
                    math.ceil(len(train_images) / args.train_batch_size),
                    total_losses, train_accuracy.result()), end='')

                step += 1
                count += 1

            # Save models after epoch
            if not os.path.exists(os.path.join(args.save_path, args.backbone_model + '_' + args.loss_head_type)):
                os.makedirs(os.path.join(args.save_path, args.backbone_model + '_' + args.loss_head_type))

            model.save_weights(
                os.path.join(args.save_path, args.backbone_model + '_' + args.loss_head_type + '/epoch_{}_train_loss_{:.5f}'.format(epoch, total_losses)),
                save_format='tf')
            print("\nSaved checkpoint for epoch {} step {}".format(epoch + 1, step))

        test_step([test_images, test_labels])

    else:
        with strategy.scope():
            # Start training
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            mc_callback = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(args.save_path, args.backbone_model + '_' + args.loss_head_type + '/epoch_{epoch}.ckpt'),
                verbose=1,
                save_weights_only=True)

            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=train_log_dir,
                update_freq=args.train_batch_size)

            callbacks = [mc_callback, tb_callback]

            history = model.fit([train_images, train_labels], train_labels,
                                epochs=args.max_epoch,
                                batch_size=args.train_batch_size,
                                callbacks=callbacks)

            # Test
            test_step([test_images, test_labels])

