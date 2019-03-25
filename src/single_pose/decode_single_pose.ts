import * as tf from '@tensorflow/tfjs';

import {partNames} from '../keypoints';
import {OutputStride} from '../mobilenet';
import {Keypoint, Pose} from '../types';
import {toTensorBuffer} from '../util';

import {argmax2d} from './argmax2d';
import {getOffsetPoints, getPointsConfidence} from './util';

export async function decodeSinglePose(
    heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D,
    outputStride: OutputStride): Promise<Pose> {
  let totalScore = 0.0;

  const heatmapValues = argmax2d(heatmapScores);

  const [scoresBuffer, offsetsBuffer, heatmapValuesBuffer] = await Promise.all([
    toTensorBuffer(heatmapScores), toTensorBuffer(offsets),
    toTensorBuffer(heatmapValues, 'int32')
  ]);

  const offsetPoints =
      getOffsetPoints(heatmapValuesBuffer, outputStride, offsetsBuffer);
  const offsetPointsBuffer = await toTensorBuffer(offsetPoints);

  const keypointConfidence =
      Array.from(getPointsConfidence(scoresBuffer, heatmapValuesBuffer));

  const keypoints = keypointConfidence.map((score, keypointId): Keypoint => {
    totalScore += score;
    return {
      position: {
        y: offsetPointsBuffer.get(keypointId, 0),
        x: offsetPointsBuffer.get(keypointId, 1)
      },
      part: partNames[keypointId],
      score
    };
  });

  heatmapValues.dispose();
  offsetPoints.dispose();

  return {keypoints, score: totalScore / keypoints.length};
}
