
import tensorflow as tf

def tensor_size(tensor):
  s = tf.shape(tensor)
  return tf.reduce_prod(s[1:])


def loss_style(gram_source, gram_target, weight, bs):
  style_loss = []
  for layer in gram_source:
    sz = tf.dtypes.cast(
      tensor_size(gram_source[layer]), 'float32')
    style_loss.append(2 * tf.nn.l2_loss(
                      gram_source[layer] -
                      gram_target[layer]) /
                      sz)
  style_loss = (weight * tf.reduce_sum(style_loss) / bs)
  return style_loss


def loss_content(content_source, content_target, weight, bs):
  content_loss = []
  for layer in content_source:
    sz = tf.dtypes.cast(
      tensor_size(content_source[layer]), 'float32')

    content_loss.append(2 * tf.nn.l2_loss(
                        content_source[layer] - 
                        content_target[layer]) /
                        sz)
  content_loss = (weight * tf.reduce_sum(content_loss) / bs)
  return content_loss


def loss_tv(output, weight, bs):
  # tv_loss = tf.reduce_sum(output)
  tv_y_size = tf.dtypes.cast(tensor_size(output[:, 1:, :, :]), 'float32')
  tv_x_size = tf.dtypes.cast(tensor_size(output[:, :, 1:, :]), 'float32')
  shape = tf.shape(output)
  y_tv = tf.nn.l2_loss(output[:, 1:, :, :] -
                       output[:, :shape[1] - 1, :, :])
  x_tv = tf.nn.l2_loss(output[:, :, 1:, :] -
                       output[:, :, :shape[2] - 1, :])  
  tv_loss = (weight * 2 *
             (x_tv / tv_x_size + y_tv / tv_y_size) /
             bs)
  return tv_loss