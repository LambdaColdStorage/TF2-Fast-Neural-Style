import os
import argparse
import datetime
import numpy as np
from PIL import Image

import tensorflow as tf

import net
import data
import loss
import vgg_preprocessing
from vgg_preprocessing import preprocess_train

# Content layer where will pull our feature maps
CONTENT_LAYERS = ['block4_conv2'] 

# Style layer we are interested in
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

STYLE_IMAGE_SIZE = 512
TEST_IMAGE_SIZE = 1024

def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("mode",
                      help="Mode to run the script",
                      choices=["train", "infer"])
  parser.add_argument("--style_image_path",
                      help="Directory to save mode",
                      type=str,
                      default="")
  parser.add_argument("--train_csv_path",
                      help="CSV file that contains paths of training images",
                      type=str,
                      default="")
  parser.add_argument("--test_images_path",
                      help="A comma seperated string for paths of testing images",
                      type=str,
                      default="")
  parser.add_argument("--style_w",
                      help="Weight for style loss",
                      type=float,
                      default=100.0)
  parser.add_argument("--content_w",
                      help="Weight for content loss",
                      type=float,
                      default=15.0)  
  parser.add_argument("--tv_w",
                      help="Weight for total variation loss",
                      type=float,
                      default=200.0)
  parser.add_argument("--bs_per_gpu",
                      help="batch size on each GPU",
                      type=int,
                      default=4)
  parser.add_argument("--num_epochs",
                      help="Number of training epochs",
                      type=int,
                      default=10)
  parser.add_argument("--decay_epoch",
                      help="Epoch index for learning rate decay",
                      type=str,
                      default='8,10')
  parser.add_argument("--base_learning_rate",
                      help="Base learning rate",
                      type=float,
                      default=0.002)

  args = parser.parse_args()

  STYLE_NAME = os.path.basename(args.style_image_path).split('.')[0]
  
  # Test images
  test_img = {}
  for path in args.test_images_path.split(','):
    name = os.path.basename(path).split('.')[0]
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img)    
    img = tf.cast(img, tf.float32)
    img = vgg_preprocessing._mean_image_subtraction(img)
    img = vgg_preprocessing._aspect_preserving_resize(img, TEST_IMAGE_SIZE)
    img = tf.expand_dims(img, 0)
    test_img[name] = img

  input_shape = (None, None, 3)
  img_input = tf.keras.layers.Input(shape=input_shape)

  output = net.style_net(img_input)
  model_output = tf.keras.models.Model(img_input, output)


  if args.mode == "infer":
    model_output.load_weights('model/' + STYLE_NAME + '_model.h5')
    for key in test_img:
      x = test_img[key]
      img = model_output.predict(x)[0]
      img = np.ndarray.astype(img, np.uint8)
      img = Image.fromarray(img, 'RGB')
      img.show()
      try:
          os.stat("output")
      except:
          os.makedirs("output")       
      img.save("output/" + key + ".jpg", "JPEG")    
  elif args.mode == "train":
    # Style imagse 
    style_img = tf.io.read_file(args.style_image_path)
    style_img = tf.image.decode_image(style_img)
    style_img = vgg_preprocessing._aspect_preserving_resize(style_img, STYLE_IMAGE_SIZE)
    style_img = tf.cast(style_img, tf.float32)
    style_img = vgg_preprocessing._mean_image_subtraction(style_img)
    style_img = tf.expand_dims(style_img, 0)

    # Training images
    train_images_path = data.load_csv(args.train_csv_path)
    NUM_TRAIN_SAMPLES = len(train_images_path)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_path))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).map(preprocess_train).batch(args.bs_per_gpu, drop_remainder=True)

    # Backbone for losses
    backbone = tf.keras.applications.vgg19.VGG19(
      include_top=False, weights='imagenet')
    backbone.trainable = False

    # Network for Gram matrices
    gram = {}
    for layer in STYLE_LAYERS:
      gram[layer] = net.compute_gram(backbone.get_layer(layer).output)
    model_gram = tf.keras.models.Model(
      backbone.input, gram, name = 'model_gram')

    # Network for content features
    content = {}
    for layer in CONTENT_LAYERS:
      content[layer] = backbone.get_layer(layer).output
    model_content = tf.keras.models.Model(
      backbone.input, content, name = 'model_content')

    # Pixel values of the stylized image are between [0, 255]. 
    # Preprocess before feeding into VGG 
    output_mean_subtracted = vgg_preprocessing._mean_image_subtraction(
          output)

    # Source and target for computing loss
    content_source = model_content(output_mean_subtracted)
    content_target = model_content(img_input)
    gram_source = model_gram(output_mean_subtracted)
    gram_target = {}
    gt = model_gram.predict(style_img)
    idx = 0
    for layer in STYLE_LAYERS:
      gram_target[layer] = gt[idx]
      idx = idx + 1

    # Compute losses
    loss_s = loss.loss_style(gram_source, gram_target, args.style_w, args.bs_per_gpu)
    loss_c = loss.loss_content(content_source, content_target, args.content_w, args.bs_per_gpu)
    loss_tv = loss.loss_tv(output, args.tv_w, args.bs_per_gpu)

    def loss_total(y_true, y_pred):
      # Hack: ignore Keras's default input for loss function.
      # Add global variables for total loss
      loss_total = loss_s + loss_c + loss_tv
      return loss_total

    opt = tf.keras.optimizers.RMSprop()

    model_output.compile(
      loss = loss_total,
      optimizer=opt)

    args.decay_epoch = [int(x) for x in args.decay_epoch.split(',')]
    for i in range(len(args.decay_epoch)):
      args.decay_epoch[i] = (pow(0.1, i + 1), args.decay_epoch[i])

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def schedule(epoch):
      initial_learning_rate = args.base_learning_rate
      learning_rate = initial_learning_rate
      for mult, start_epoch in args.decay_epoch:
        if epoch >= start_epoch:
          learning_rate = initial_learning_rate * mult
        else:
          break
      tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
      return learning_rate

    class MyCustomCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        try:
            os.stat("val/" + STYLE_NAME + "_" + time_stamp + "/epoch" + str(epoch))
        except:
            os.makedirs("val/" + STYLE_NAME+ "_" + time_stamp + "/epoch" + str(epoch)) 
        for key in test_img:
          img = model_output.predict(test_img[key])[0]
          img = np.ndarray.astype(img, np.uint8)
          img = Image.fromarray(img, 'RGB')
          img.save("val/" + STYLE_NAME+ "_" + time_stamp + "/epoch" + str(epoch) + "/" + key + ".jpg", "JPEG")

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)
    test_callback = MyCustomCallback()
    log_dir="logs/fit/" + time_stamp
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      update_freq='batch',
      histogram_freq=1)

    model_output.fit(
      train_dataset,
      epochs=args.num_epochs,
      callbacks=[lr_schedule_callback, test_callback, tensorboard_callback])
    model_output.save_weights('model/' + STYLE_NAME + '_model.h5')


if __name__ == "__main__":
  main()
