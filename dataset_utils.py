import tensorflow as tf


def get_mnist_dataset(BATCH_SIZE=256, BUFFER_SIZE=60000):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    mnist_train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return mnist_train_dataset


if __name__ == "__main__":
    mnist_train_dataset = get_mnist_dataset(BATCH_SIZE=256, BUFFER_SIZE=60000)
    for batch_image in mnist_train_dataset.take(3):
        print(batch_image.shape)
