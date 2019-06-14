import tensorflow as tf

para_conv = [{'filters': 32,
              'kernel_size': [8, 8],
              'strides': [1, 1]},
             {'filters': 64,
              'kernel_size': [4, 4],
              'strides': [2, 2]},
             {'filters': 128,
              'kernel_size': [4, 4],
              'strides': [2, 2]}]

para_deconv = [{'filters': 64,
                'kernel_size': [4, 4],
                'strides': [2, 2]},
               {'filters': 32,
                'kernel_size': [4, 4],
                'strides': [2, 2]},
               {'filters': 3,
                'kernel_size': [8, 8],
                'strides': [1, 1]}]

para_res = {'filters': 128,
            'kernel_size': [4, 4],
            'strides': [1, 1]}


def instance_norm_layer(inputs):
  mu, sigma_sq = tf.compat.v1.nn.moments(inputs, [1, 2], keep_dims=True)
  epsilon = 1e-3

  outputs = (inputs - mu) / (sigma_sq + epsilon)**(.5)
  return outputs


def residual_layer(inputs, para):

  outputs = tf.keras.layers.Conv2D(
    para['filters'],
    (para['kernel_size'][0], para['kernel_size'][1]),
    para['strides'],
    use_bias=False,
    padding='SAME')(inputs)
  outputs = instance_norm_layer(outputs)
  outputs = tf.keras.layers.Activation('relu')(outputs)

  outputs = tf.keras.layers.Conv2D(
    para['filters'],
    (para['kernel_size'][0], para['kernel_size'][1]),
    para['strides'],
    use_bias=False,
    padding='SAME')(outputs)
  outputs = instance_norm_layer(outputs)

  return inputs + outputs


def style_net(img_input):

  # block 1
  output = tf.keras.layers.Conv2D(
    para_conv[0]['filters'],
    (para_conv[0]['kernel_size'][0], para_conv[0]['kernel_size'][1]),
    para_conv[0]['strides'],
    use_bias=False,
    padding='SAME')(img_input)
  output = instance_norm_layer(output)
  output = tf.keras.layers.Activation('relu')(output)

  # block 2
  output = tf.keras.layers.Conv2D(
    para_conv[1]['filters'],
    (para_conv[1]['kernel_size'][0], para_conv[1]['kernel_size'][1]),
    para_conv[1]['strides'],
    use_bias=False,
    padding='SAME')(output)
  output = instance_norm_layer(output)
  output = tf.keras.layers.Activation('relu')(output)

  # block 3
  output = tf.keras.layers.Conv2D(
    para_conv[2]['filters'],
    (para_conv[2]['kernel_size'][0], para_conv[2]['kernel_size'][1]),
    para_conv[2]['strides'],
    use_bias=False,
    padding='SAME')(output)
  output = instance_norm_layer(output)
  output = tf.keras.layers.Activation('relu')(output)

  # Residual layers
  output = residual_layer(output, para_res)
  output = residual_layer(output, para_res)
  output = residual_layer(output, para_res)
  output = residual_layer(output, para_res)
  output = residual_layer(output, para_res)


  output = tf.keras.layers.Conv2DTranspose(
    para_deconv[0]['filters'],
    (para_deconv[0]['kernel_size'][0], para_deconv[0]['kernel_size'][1]),
    para_deconv[0]['strides'],
    use_bias=False,
    padding='SAME')(output)
  output = instance_norm_layer(output)
  output = tf.keras.layers.Activation('relu')(output)

  output = tf.keras.layers.Conv2DTranspose(
    para_deconv[1]['filters'],
    (para_deconv[1]['kernel_size'][0], para_deconv[1]['kernel_size'][1]),
    para_deconv[1]['strides'],
    use_bias=False,
    padding='SAME')(output)
  output = instance_norm_layer(output)
  output = tf.keras.layers.Activation('relu')(output)

  output = tf.keras.layers.Conv2DTranspose(
    para_deconv[2]['filters'],
    (para_deconv[2]['kernel_size'][0], para_deconv[2]['kernel_size'][1]),
    para_deconv[2]['strides'],
    use_bias=False,
    padding='SAME')(output)
  output = instance_norm_layer(output)


  output = tf.keras.layers.Activation('tanh')(output) * 127.5 + 255. / 2

  return output


def compute_gram(feature, data_format='channels_last'):
    layer_shape = tf.shape(feature)
    bs = layer_shape[0]
    height = (layer_shape[1] if data_format == 'channels_last'
              else layer_shape[2])
    width = (layer_shape[2] if data_format == 'channels_last'
             else layer_shape[3])
    filters = (layer_shape[3] if data_format == 'channels_last'
               else layer_shape[1])
    size = height * width * filters
    feats = (tf.reshape(feature, (bs, height * width, filters))
             if data_format == 'channels_last'
             else tf.reshape(feature, (bs, filters, height * width)))
    feats_T = tf.transpose(feats, perm=[0, 2, 1])
    gram = (tf.matmul(feats_T, feats) / tf.cast(size, tf.float32)
            if data_format == 'channels_last'
            else tf.matmul(feats, feats_T) / tf.cast(size, tf.float32))
    return gram
