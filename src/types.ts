import * as tf from '@tensorflow/tfjs';

export type Vector2D = {
  y: number,
  x: number
};

export type Part = {
  heatmapX: number,
  heatmapY: number,
  id: number
};

export type PartWithScore = {
  score: number,
  part: Part
};

export type Keypoint = {
  score: number,
  position: Vector2D,
  part: string
};

export type Pose = {
  keypoints: Keypoint[],
  score: number,
};

export type PosenetInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;

export type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;
