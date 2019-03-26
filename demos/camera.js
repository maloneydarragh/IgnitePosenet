import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, drawPersonTag, drawSkeleton} from './demo_util';
import toastr from 'toastr';

const videoWidth = 1200;
const videoHeight = 800;
var leftShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
var rightShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const stats = new Stats();
toastr.options = {
    "closeButton": false,
    "debug": false,
    "newestOnTop": false,
    "progressBar": false,
    "positionClass": "toast-top-right",
    "preventDuplicates": false,
    "onclick": null,
    "showDuration": "300",
    "hideDuration": "1000",
    "timeOut": "5000",
    "extendedTimeOut": "1000",
    "showEasing": "swing",
    "hideEasing": "linear",
    "showMethod": "fadeIn",
    "hideMethod": "fadeOut"
};

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
    guiState.net = net;

    if (cameras.length > 0) {
        guiState.camera = cameras[0].deviceId;
    }
}


var personNoseXArray = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];


/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;

    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
            video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(
            video, imageScaleFactor, flipHorizontal, outputStride,
            guiState.multiPoseDetection.maxPoseDetections,
            guiState.multiPoseDetection.minPartConfidence,
            guiState.multiPoseDetection.nmsRadius);
            //console.log('Poses:', poses);
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

      //console.log("************** POSES: *******************");
     // console.log("****  poses count: " + poses.length);
      //console.log(poses);

      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores

      var  index = 0;

      //array of colors, so each person has a different color
      var colors = ['orange','purple','blue','green','white','pink','brown','black'];


      /*
         need to keep track of individuals, so need to keep track of x position of certain keypoints
      */
      var personArray = [];

      poses.forEach(({score, keypoints}) => {

          if (score >= minPoseConfidence) {
              drawPersonTag(colors[index],keypoints, ctx, index);
              if (guiState.output.showPoints) {
                  drawKeypoints(keypoints, minPartConfidence, ctx);
              }
              if (guiState.output.showSkeleton) {

                  //for each x co-ordinate, check against existing values
                  var currentNoseX = keypoints[0].position.x;  //should be nose

                  //if first time value stored
                  if (personNoseXArray[index] === 0){
                      personNoseXArray[index] = currentNoseX;
                      drawSkeleton(colors[index],keypoints, minPartConfidence, ctx);
                      console.log("**  NEW ***: " + currentNoseX);
                  }else{
                      if (checkIfValueWithinRange(currentNoseX, personNoseXArray[index], 50)){
                          console.log("**  LESS THAN RANGE ***: " + personNoseXArray[index] + " , " + currentNoseX +  " , " + Math.abs(personNoseXArray[index] - currentNoseX));
                          personNoseXArray[index] = currentNoseX;

                          //check if a keypoint's position is lower than a percentage (tbc), if so draw big red lines
                          if (checkIfSomeoneHasFallen(keypoints)){
                              drawSkeleton(colors[index],keypoints, minPartConfidence, ctx);
                          }else{
                              drawSkeleton('red',keypoints, minPartConfidence, ctx);
                              // document.getElementById("alertAudio").play();
                              leftShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                              rightShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                          }


                      }else{
                          console.log("**  GREATER THAN RANGE ***: " + personNoseXArray[index] + " , " + currentNoseX +  " , " + Math.abs(personNoseXArray[index] - currentNoseX));
                          personNoseXArray[index] = currentNoseX;
                          //drawSkeleton('white',keypoints, minPartConfidence, ctx);
                          //check if a keypoint's position is lower than a percentage (tbc), if so draw big red lines
                          if (checkIfSomeoneHasFallen(keypoints)){
                              drawSkeleton(colors[index],keypoints, minPartConfidence, ctx);
                          }else{
                              drawSkeleton('red',keypoints, minPartConfidence, ctx);
                              // document.getElementById("alertAudio").play();
                              leftShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                              rightShoulderArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                          }
                      }
                  }




                  //drawSkeleton(colorrr,keypoints, minPartConfidence, ctx);
                  index++;



              }
              if (guiState.output.showBoundingBox) {
                  drawBoundingBox(keypoints, ctx);
              }
          }
      });

      // End monitoring code for frames per second
      stats.end();

      requestAnimationFrame(poseDetectionFrame);
  }

    poseDetectionFrame();
}

//initial value
var nosePosition = 0;

//check if a keypoint's position is lower than a percentage (tbc), if so draw big red lines
function checkIfSomeoneHasFallen(keypoints){
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    if(leftShoulder.score > 0.1) {
        // console.log('left shoulder is detected');
        leftShoulderArray = leftShoulderArray.slice(1);
        leftShoulderArray = leftShoulderArray.concat(leftShoulder.position.y);
        if(leftShoulderArray[9]-leftShoulderArray[0]>200 && leftShoulderArray[0] !== 0){
            return false;
        }
    }
    else if(rightShoulder.score > 0.1) {
        // console.log('right shoulder is detected');
        rightShoulderArray = rightShoulderArray.slice(1);
        rightShoulderArray = rightShoulderArray.concat(rightShoulder.position.y);
        if(rightShoulderArray[9]-rightShoulderArray[0]>200 && rightShoulderArray[0] !==0){
            return false;
        }
    }

    return true;
}

function checkIfValueWithinRange(firstValue, secondValue, range){
    if ((Math.abs(firstValue - secondValue) < range)) {
        return true;
    }else{
        return false;
    }
}


/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
