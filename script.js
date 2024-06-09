let amountTrainData = 100;
let rangeTrainData = 4;
let amountLayers = 2;
let activationFunction = 'relu';
let data;
let model;
let amountEpochs = 50;
let optimizer = 'adam';
let learnRate = 0.01;
let xInput = 0;
let tensorDataTrainClean;
let tensorDataTestClean;
let noiseVariance = 0.05;

function getRandomNumber(range){
    const randomNumber = (Math.random() * range) - (range * 0.5);
    return randomNumber;
}

function calculateFunctionResult(x){
    return (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6);
}

function getData(amount, range){
    let data = [];
    for(var i = 0; i < amount; i++){
        let randomX = getRandomNumber(range);
        let randomY = calculateFunctionResult(randomX);
        let noiseY = randomY + tf.randomNormal([1], 0, Math.sqrt(noiseVariance)).dataSync()[0];
        data.push({x: randomX, y: randomY, yNoise: noiseY});
    }
    return data;
}

function setup(){
  // Create the model
  model = createModel(amountLayers, activationFunction);
  tfvis.show.modelSummary(
    document.getElementById('modelData'), 
    model
  );

  const layersSlider = document.getElementById('numberLayers');
  const layersLabel = document.getElementById('numberLayersLabel');
  const activationSelection = document.getElementById('activationFunction');
  const epochSlider = document.getElementById('numberEpochs');
  const epochLabel = document.getElementById('numberEpochsLabel');
  const optimizerSelection = document.getElementById('optimizer');
  const learnRateSlider = document.getElementById('learnRate');
  const learnRateLabel = document.getElementById('learnRateLabel');
 

  activationSelection.value = activationFunction;


  data = getData(amountTrainData, rangeTrainData);

  // split data in test and training data
  const N = data.length;
  const halfN = Math.floor(N / 2);
  // Daten in Trainings- und Testdatensätze aufteilen
  const indices = tf.util.createShuffledIndices(N);
  const trainIndices = indices.slice(0, halfN);
  const testIndices = indices.slice(halfN);

  const xTrain = [];
  const yTrain = [];
  const yTrainNoise = [];
  trainIndices.forEach(i => {
    xTrain.push(data[i].x);
    yTrain.push(data[i].y);
    yTrainNoise.push(data[i].yNoise);
  });

  const trainDataClean = [];
  xTrain.forEach((x, idx) => {
    trainDataClean.push({
      x: x, 
      y: yTrain[idx]
    });
  });
  tensorDataTrainClean = convertToTensor(trainDataClean);

  const xTest = [];
  const yTest = [];
  const yTestNoise = [];
  testIndices.forEach(i => {
    xTest.push(data[i].x);
    yTest.push(data[i].y);
    yTestNoise.push(data[i].yNoise);
  });

  const testDataClean = [];
  xTest.forEach((x, idx) => {
    testDataClean.push({
      x: x, 
      y: yTest[idx]
    });
  });
  tensorDataTestClean = convertToTensor(testDataClean);




  renderTrainDataPlot(xTrain, yTrain, xTest, yTest);
  renderTrainDataPlotWithNoise(xTrain, yTrainNoise, xTest, yTestNoise);


  learnRateSlider.value = learnRate;
  learnRateLabel.innerHTML = "Learnrate: " + learnRate;

  epochSlider.value = amountEpochs;
  epochLabel.innerHTML = "Amount of Epochs: " + amountEpochs;

  layersSlider.value = amountLayers;
  layersLabel.innerHTML = "Amount of hidden Layers: " + amountLayers; 

  
  optimizerSelection.value = optimizer;
  optimizerSelection.oninput = function(){
    optimizer = optimizerSelection.value;
    run(model, amountEpochs, optimizer, learnRate);
  }

  learnRateSlider.oninput = function(){
    learnRate = parseFloat(learnRateSlider.value);
    learnRateLabel.innerHTML = "Learnrate: " + learnRate;
    run(model, amountEpochs, optimizer, learnRate);
  }

  activationSelection.oninput = function(){
    activationFunction = activationSelection.value;
    run(model, amountEpochs, optimizer, learnRate);
    model.layers.forEach(layer => {
      layer.dispose()
    });
    model = createModel(amountLayers, activationFunction);
    tfvis.show.modelSummary(
      document.getElementById('modelData'), 
      model
    );
  }

  epochSlider.oninput = function(){
    amountEpochs = parseInt(epochSlider.value);
    epochLabel.innerHTML = "Amount of Epochs: " + amountEpochs;
    run(model, amountEpochs, optimizer, learnRate);
  }

  layersSlider.oninput = function(){
    amountLayers = parseInt(layersSlider.value);
    layersLabel.innerHTML = "Amount of hidden Layers: " + amountLayers;
    model.layers.forEach(layer => {
      layer.dispose()
    });
    model = createModel(amountLayers, activationFunction);
    tfvis.show.modelSummary(
      document.getElementById('modelData'), 
      model
    );
    run(model, amountEpochs, optimizer, learnRate);
  }

  run(model, amountEpochs, optimizer, learnRate);
  
}

function renderTrainDataPlot(xTrain, yTrain, xTest, yTest){
  const trainDataSeries = xTrain.map((x, i) => ({ x: x, y: yTrain[i] }));
  const testDataSeries = xTest.map((x, i) => ({ x: x, y: yTest[i] }));
  const series =  ['Train', 'Test'];
  tfvis.render.scatterplot(
    document.getElementById('inputDataPlot'),
    {values: [trainDataSeries, testDataSeries], series},
    {
        xLabel: 'x',
        yLabel: 'y',
        height: 600
    }
  );
}

function renderTrainDataPlotWithNoise(xTrain, yTrainNoise, xTest, yTestNoise){
  const trainDataSeries = xTrain.map((x, i) => ({ x: x, y: yTrainNoise[i] }));
  const testDataSeries = xTest.map((x, i) => ({ x: x, y: yTestNoise[i] }));
  const series =  ['Train', 'Test'];
  tfvis.render.scatterplot(
    document.getElementById('inputDataPlotWithNoise'),
    {values: [trainDataSeries, testDataSeries], series},
    {
        xLabel: 'x',
        yLabel: 'y',
        height: 600
    }
  );
}

async function run(currentModel, numberEpochs, currentOptimizer, currentLearnRate){
  // Convert the data to a form we can use for training.
  
  const {inputs, labels} = tensorDataTrainClean;
  const inputTestData = tensorDataTestClean.inputs;
  const labelsTestData = tensorDataTestClean.labels;

  // Train the model with train data
  await trainModel(currentModel, inputs, labels, numberEpochs, currentOptimizer, currentLearnRate, 'trainDataPlot');
  console.log('Done Training');
  // Make some predictions using the model and compare them to the
  // train data
  testModel(currentModel, data, tensorDataTrainClean, 'resultPlotTrainData');
  // testData

  // Train the model with test data
  await trainModel(currentModel, inputTestData, labelsTestData, numberEpochs, currentOptimizer, currentLearnRate, 'testDataPlot');
  console.log('Done Training (test data)');
  testModel(currentModel, data, tensorDataTestClean, 'resultPlotTestData');
}

function createModel(amountLayer, activationFunction) {
    // Create a sequential model
    const model = tf.sequential();

    // // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    for(let i = 0; i < amountLayer; i++){
      model.add(tf.layers.dense({units: 50, activation: activationFunction}));
    }

    // Add an output layer
    model.add(tf.layers.dense({units: 1}));
  
    return model;
  }

  /**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
  
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.x)
      const labels = data.map(d => d.y);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

async function trainModel(currentModel, inputs, labels, numberEpochs, currentOptimizer, currentLearnRate, containerName) {
    // Prepare the model for training.

    let opti;
    switch(currentOptimizer){
      case 'sgd':
        opti = tf.train.sgd(currentLearnRate);
        break;
      case 'adam':
        opti = tf.train.adam(currentLearnRate);
        break;
      case 'momentum':
        opti = tf.train.momentum(currentLearnRate);
        break;
      case 'adagrad':
        opti = tf.train.adagrad(currentLearnRate);
        break;
      case 'adadelta':
        opti = tf.train.adadelta(currentLearnRate);
        break;
      case 'adamax':
        opti = tf.train.adamax(currentLearnRate);
        break;
      default:
        opti = tf.train.adam(currentLearnRate);

    }
    console.log(opti);
    currentModel.compile({
        optimizer: opti,
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });
    const batchSize = 32;
    const epochs = numberEpochs;


    return await currentModel.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
        document.getElementById(containerName),
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(currentModel, inputData, normalizationData, containerName) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xs = tf.linspace(0, 1, 100);
      const preds = currentModel.predict(xs.reshape([100, 1]));
  
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.x, y: d.y,
    }));
  
  
    tfvis.render.scatterplot(
      document.getElementById(containerName),
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLabel: 'x',
        yLabel: 'y',
        height: 500
      }
    );
  }

document.addEventListener('DOMContentLoaded', setup());