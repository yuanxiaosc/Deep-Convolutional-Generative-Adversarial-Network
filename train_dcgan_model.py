import os
import time
import glob
import imageio

import matplotlib.pyplot as plt
import tensorflow as tf

from dcgan_model import Generator, Discriminator, discriminator_loss, generator_loss
from dataset_utils import get_mnist_dataset


def generate_and_save_images(model, epoch, test_input, store_produce_image_dir):
    if not os.path.exists(store_produce_image_dir):
        os.mkdir(store_produce_image_dir)

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    save_image_path = os.path.join(store_produce_image_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(save_image_path)
    plt.show()


def images_to_gif(anim_file='dcgan.gif', store_produce_image_dir=""):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(store_produce_image_dir + '/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def train_dcgan_main(BATCH_SIZE, EPOCHS, noise_dim, num_examples_to_generate, checkpoint_dir, store_produce_image_dir):
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed, store_produce_image_dir)

            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, seed, store_produce_image_dir)

    # prepare data
    train_dataset = get_mnist_dataset(BATCH_SIZE, BUFFER_SIZE=60000)

    # create model
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)



    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # train model
    train(train_dataset, EPOCHS)

    # produce images to gif file
    images_to_gif(anim_file='dcgan.gif', store_produce_image_dir=store_produce_image_dir)


if __name__ == "__main__":
    BATCH_SIZE = 256
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    checkpoint_dir = './training_checkpoints'
    store_produce_image_dir = "train_epochs_produce_images"
    train_dcgan_main(BATCH_SIZE, EPOCHS, noise_dim, num_examples_to_generate, checkpoint_dir, store_produce_image_dir)
