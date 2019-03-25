import * as tf from '@tensorflow/tfjs';

import {CheckpointLoader} from './checkpoint_loader';
import {checkpoints} from './checkpoints';
import {assertValidOutputStride, assertValidScaleFactor, MobileNet, MobileNetMultiplier, OutputStride} from './mobilenet';
import {ModelWeights} from './model_weights';
import {decodeMultiplePoses} from './multi_pose/decode_multiple_poses';
import {decodeSinglePose} from './single_pose/decode_single_pose';
import {Pose, PosenetInput} from './types';
import {getInputTensorDimensions, getValidResolution, scalePose, scalePoses, toResizedInputTensor} from './util';

export type PoseNetResolution = 161|193|257|289|321|353|385|417|449|481|513;

export class PoseNet {
  mobileNet: MobileNet;

  constructor(mobileNet: MobileNet) {
    this.mobileNet = mobileNet;
  }

  predictForSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 16):
      {heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D} {
    assertValidOutputStride(outputStride);
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      const heatmaps =
          this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');

      const offsets = this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');

      return {heatmapScores: heatmaps.sigmoid(), offsets};
    });
  }

  predictForMultiPose(input: tf.Tensor3D, outputStride: OutputStride = 16): {
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D
  } {
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      const heatmaps =
          this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');

      const offsets = this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');

      const displacementFwd =
          this.mobileNet.convToOutput(mobileNetOutput, 'displacement_fwd_2');

      const displacementBwd =
          this.mobileNet.convToOutput(mobileNetOutput, 'displacement_bwd_2');

      return {
        heatmapScores: heatmaps.sigmoid(),
        offsets,
        displacementFwd,
        displacementBwd
      };
    });
  }

  async estimateSinglePose(
      input: PosenetInput, imageScaleFactor = 0.5, flipHorizontal = false,
      outputStride: OutputStride = 16): Promise<Pose> {
    assertValidOutputStride(outputStride);
    assertValidScaleFactor(imageScaleFactor);

    const [height, width] = getInputTensorDimensions(input);

    const resizedHeight =
        getValidResolution(imageScaleFactor, height, outputStride);
    const resizedWidth =
        getValidResolution(imageScaleFactor, width, outputStride);

    const {heatmapScores, offsets} = tf.tidy(() => {
      const inputTensor = toResizedInputTensor(
          input, resizedHeight, resizedWidth, flipHorizontal);

      return this.predictForSinglePose(inputTensor, outputStride);
    });

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);

    const scaleY = height / resizedHeight;
    const scaleX = width / resizedWidth;

    heatmapScores.dispose();
    offsets.dispose();

    return scalePose(pose, scaleY, scaleX);
  }

  async estimateMultiplePoses(
      input: PosenetInput, imageScaleFactor = 0.5, flipHorizontal = false,
      outputStride: OutputStride = 16, maxDetections = 5, scoreThreshold = .5,
      nmsRadius = 20): Promise<Pose[]> {
    assertValidOutputStride(outputStride);
    assertValidScaleFactor(imageScaleFactor);

    const [height, width] = getInputTensorDimensions(input);
    const resizedHeight =
        getValidResolution(imageScaleFactor, height, outputStride);
    const resizedWidth =
        getValidResolution(imageScaleFactor, width, outputStride);

    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        tf.tidy(() => {
          const inputTensor = toResizedInputTensor(
              input, resizedHeight, resizedWidth, flipHorizontal);
          return this.predictForMultiPose(inputTensor, outputStride);
        });

    const poses = await decodeMultiplePoses(
        heatmapScores, offsets, displacementFwd, displacementBwd, outputStride,
        maxDetections, scoreThreshold, nmsRadius);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    const scaleY = height / resizedHeight;
    const scaleX = width / resizedWidth;

    return scalePoses(poses, scaleY, scaleX);
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

export async function load(multiplier: MobileNetMultiplier = 1.01):
    Promise<PoseNet> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  // TODO: figure out better way to decide below.
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'number',
      () => `got multiplier type of ${typeof multiplier} when it should be a ` +
          `number.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier.toString()) >= 0,
      () => `invalid multiplier value of ${
                multiplier}.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  const mobileNet: MobileNet = await mobilenetLoader.load(multiplier);

  return new PoseNet(mobileNet);
}

export const mobilenetLoader = {
  load: async(multiplier: MobileNetMultiplier): Promise<MobileNet> => {
    const checkpoint = checkpoints[multiplier];

    const checkpointLoader = new CheckpointLoader(checkpoint.url);

    const variables = await checkpointLoader.getAllVariables();

    const weights = new ModelWeights(variables);

    return new MobileNet(weights, checkpoint.architecture);
  },
};
