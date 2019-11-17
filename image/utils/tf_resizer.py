import tensorflow as tf
import pandas as pd

def tf_resizer(inp_img_row, scale_size, dims):
    img = tf.image.decode_jpeg(tf.io.read_file(inp_img_row.id), channels=3)
    if scale_size != 1:
        img = tf.image.resize(img, size=(tf.cast(round(inp_img_row.height*scale_size), tf.int32),
                                         tf.cast(round(inp_img_row.width*scale_size), tf.int32)),
                                    preserve_aspect_ratio=True)
    elif dims[0] != None:
        img = tf.image.resize(img, [dims[0], dims[1]])
    img = tf.cast(img, tf.uint8)
    img = tf.image.encode_jpeg(img, quality=95, format='rgb', optimize_size=True)
    new_path = inp_img_row.id.split("train")
    tf.io.write_file(new_path[0]+"train_temp"+new_path[1], img)
    return None
