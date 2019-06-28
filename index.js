import * as tf from "@tensorflow/tfjs";
import { MnistData } from "./data";

import "bootstrap/dist/css/bootstrap.css";
import "babel-polyfill";

// This variable will hold the model
let model;

//Helper - Logging to console
function createLogEntry(entry) {
  document.getElementById("log").innerHTML += "<br>" + entry;
}

function createModel() {
  createLogEntry("Creating model");
  model = tf.sequential();
  createLogEntry("Model created");

  createLogEntry("Adding layers");

  // 1. 2D Convolutional layer
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1], // Representative of the pixel shape of the input images
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "VarianceScaling"
    })
  );

  // 2. 2D Max pooling layer
  // Down sample the image so it's half the size of the input from the previous layer
  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2], // Size of the sliding windows (2px x 2px)
      strides: [2, 2] // How many px the filter window slides over the input image
    })
  );

  // 3. Repeat the convolutional layer
  // Input shape not specified here since it is determined by the output of the previous layer
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "VarianceScaling"
    })
  );

  // 4. Repeat the max pooling layer
  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  // 5. Flatten layer
  // This flattens the output from the previous layer to a vector
  model.add(tf.layers.flatten()); 

  // 6. Dense layer
  //
  model.add(
    tf.layers.dense({
      units: 10, // Since we are doing a 10-class classification (0 to 9)
      kernelInitializer: "VarianceScaling",
      activation: "softmax" // Creates a probability dist. over the 10 classes
    })
  );

  createLogEntry("Layers created");

  createLogEntry("Compiling the model");
  model.compile({
    optimizer: tf.train.sgd(0.15),
    loss: "categoricalCrossentropy"
  });
  createLogEntry("Model compiled!");
}

// Variable to hold the MNIST data
let data;

// Function that async loads the MNIST data
async function load() {
  createLogEntry("Loading MNIST data");
  data = new MnistData();
  await data.load();
  createLogEntry("MNIST Data loaded successfully");
}

const BATCH_SIZE = 64; // Common to use 64 (base 2)
const TRAIN_BATCHES = 150; // Perform training in batches instead of one operation

async function train() {
  createLogEntry("Start training ...");
  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = tf.tidy(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
      return batch;
    });

    await model.fit(batch.xs, batch.labels, {
      batchSize: BATCH_SIZE,
      epochs: 2
    });

    tf.dispose(batch);

    await tf.nextFrame();
  }
  createLogEntry("Training complete");
}

// Main function that runs the neural network in proper steps
async function main() {
  createModel();
  await load();
  await train();
  document.getElementById("selectTestDataButton").disabled = false;
  document.getElementById("selectTestDataButton").innerText =
    "Pick random hand-written figure";
}

async function predict(batch) {
  tf.tidy(() => {
    const input_value = Array.from(batch.labels.argMax(1).dataSync());

    const div = document.createElement("div");
    div.className = "prediction-div";

    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

    const prediction_value = Array.from(output.argMax(1).dataSync());
    const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);

    const canvas = document.createElement("canvas");
    canvas.className = "prediction-canvas";
    draw(image.flatten(), canvas);

    const label = document.createElement("div");
    label.innerHTML = "Original Value: " + input_value;
    label.innerHTML += "<br>Prediction Value: " + prediction_value;

    if (prediction_value - input_value == 0) {
      label.innerHTML += "<br>Value recognized successfully";
    } else {
      label.innerHTML += "<br>Recognition failed!";
    }

    div.appendChild(canvas);
    div.appendChild(label);
    document.getElementById("predictionResult").appendChild(div);
  });
}

function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

document
  .getElementById("selectTestDataButton")
  .addEventListener("click", async (el, ev) => {
    const batch = data.nextTestBatch(1);
    await predict(batch);
  });

// Running the main function that creates and compiles the model
main();
