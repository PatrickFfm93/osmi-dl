const dataSetSize = 100;

var train_xs = [];
var train_ys = [];
var test_xs = [];
var test_ys = [];

for(let x = 0; x < dataSetSize; x++){
    // Set train data
    train_xs.push(x);
    let y = (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6);
    train_ys.push(y);
    
    // Set test data
    let test_x = x + Math.random();
    test_xs.push(test_x);
    
    let test_y = (test_x+0.8)*(test_x-0.2)*(test_x-0.3)*(test_x-0.6);
    test_ys.push(test_y);
}

const trainData = {X:train_xs, Y:train_ys};
const testData = {X:test_xs, Y:test_ys};


const trainTensors = {
    X: tf.tensor2d(trainData.X, [dataSetSize, 1]),
    Y: tf.tensor2d(trainData.Y, [dataSetSize, 1]) 
};

const testTensors = {
    X: tf.tensor2d(testData.X, [dataSetSize, 1]),
    Y: tf.tensor2d(testData.Y, [dataSetSize, 1])
};

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1], units: 1}));
model.compile({optimizer: 'sgd', loss: 'meanAbsoluteError'});

(async function() {
    await model.fit(trainTensors.X, trainTensors.Y, {epochs: 10});
});

model.evaluate(testTensors.X, testTensors.Y).print();