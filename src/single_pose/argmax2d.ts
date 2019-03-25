import * as tf from '@tensorflow/tfjs';

function mod(a: tf.Tensor1D, b: number): tf.Tensor1D {
  return tf.tidy(() => {
    const floored = a.div(tf.scalar(b, 'int32'));

    return a.sub(floored.mul(tf.scalar(b, 'int32')));
  });
}

export function argmax2d(inputs: tf.Tensor3D): tf.Tensor2D {
  const [height, width, depth] = inputs.shape;

  return tf.tidy(() => {
    const reshaped = inputs.reshape([height * width, depth]);
    const coords = reshaped.argMax(0) as tf.Tensor1D;

    const yCoords =
        coords.div(tf.scalar(width, 'int32')).expandDims(1) as tf.Tensor2D;
    const xCoords = mod(coords, width).expandDims(1) as tf.Tensor2D;

    return tf.concat([yCoords, xCoords], 1);
  });
}
