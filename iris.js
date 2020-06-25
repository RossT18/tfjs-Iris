"use strict";

const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const prompt = require("prompt-async");

//#region Read Datasets
const irisRaw = fs.readFileSync("iris.json", "utf8");
const irisTestsRaw = fs.readFileSync("irisTests.json", "utf8");

const iris = JSON.parse(irisRaw);
const irisTests = JSON.parse(irisTestsRaw);
//#endregion

//#region Setup training data
const trainingData = tf.tensor2d(iris.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));

const outputData = tf.tensor2d(iris.map(item => [
    item.species == "setosa" ? 1 : 0,
    item.species == "versicolor" ? 1 : 0,
    item.species == "virginica" ? 1 : 0
]));

const testingData = tf.tensor2d(irisTests.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));
//#endregion

//#region Create Model with 3 layers
const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5
}));

model.add(tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3
}));

model.add(tf.layers.dense({
    activation: "sigmoid",
    units: 3
}));

model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.06)
});
//#endregion

//#region Predict with JSON
/*
const startTime = Date.now();
model.fit(trainingData, outputData, {epochs: 100}).then((history) => {
        const endTime = Date.now();
        model.predict(testingData).print();
        console.log(history);
        console.log("Total Fit Time: " + (endTime - startTime) + "ms.");
    });
*/
//#endregion

async function convert_prompt_to_tensor2d(input) {
    return tf.tensor2d([[
        parseFloat(input.sepal_length),
        parseFloat(input.sepal_width),
        parseFloat(input.petal_length),
        parseFloat(input.petal_width)
    ]]);
}

async function getInputData() {
    prompt.start();
    const input = await prompt.get(["sepal_length", "sepal_width", "petal_length", "petal_width"]);
    const inputTensor = await convert_prompt_to_tensor2d(input);
    return inputTensor;
}

async function run(epochCount, showHistory) {
    // Train/Fit.
    const history = await model.fit(trainingData, outputData, {epochs: epochCount});

    if (showHistory) { console.log(history); }

    // Get input data.
    const input = await getInputData();
    input.print();

    // Predict.
    model.predict(input, 32, false).print();
}

run(200, true);