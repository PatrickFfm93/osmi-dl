// load the dataset data.txt
const text = "../data.txt";

// Tokenizer initialisieren und anpassen
const tokenizer = new Tokenizer();
tokenizer.fitOnTexts([text]);

// Text in Sequenzen von Token umwandeln
const sequences = tokenizer.textsToSequences([text])[0];

// Token-Index und umgekehrter Index
const wordIndex = tokenizer.wordIndex;
const reverseWordIndex = Object.keys(wordIndex).reduce((acc, key) => {
  acc[wordIndex[key]] = key;
  return acc;
}, {});

// Eingabesequenzen und Zielwerte erstellen
const sequenceLength = 5;
const examples = [];
const labels = [];

for (let i = 0; i < sequences.length - sequenceLength; i++) {
  const sequence = sequences.slice(i, i + sequenceLength);
  const label = sequences[i + sequenceLength];
  examples.push(sequence);
  labels.push(label);
}

const xs = tf.tensor2d(examples);
const ys = tf.tensor1d(labels, 'int32');

const model = tf.sequential();
model.add(tf.layers.embedding({inputDim: tokenizer.wordIndex.length + 1, outputDim: 50, inputLength: sequenceLength}));
model.add(tf.layers.lstm({units: 100, returnSequences: true}));
model.add(tf.layers.lstm({units: 100, returnSequences: false}));
model.add(tf.layers.dense({units: tokenizer.wordIndex.length + 1, activation: 'softmax'}));

model.compile({
  optimizer: tf.train.adam(0.01, batchSize = 32),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();

async function trainModel() {
    const history = await model.fit(xs, ys, {
      epochs: 50,  // Du kannst die Anzahl der Epochen anpassen
      batchSize: 32,
      callbacks: tf.callbacks.earlyStopping({monitor: 'loss'}),
    });
    console.log('Training abgeschlossen');
    console.log(history);
  }
  
  trainModel();

  function sample(preds, temperature = 1.0) {
    preds = tf.div(preds, tf.scalar(temperature));
    const expPreds = tf.exp(preds);
    const probs = expPreds.div(tf.sum(expPreds));
    const probArray = probs.dataSync();
    const cumProbs = probArray.map((p, i) => probArray.slice(0, i + 1).reduce((a, b) => a + b, 0));
    const rand = Math.random();
    return cumProbs.findIndex(p => p > rand);
  }
  
  async function generateText(startSeed, length = 20, temperature = 1.0) {
    let result = startSeed;
    let input = tokenizer.textsToSequences([startSeed])[0].slice(-sequenceLength);
  
    for (let i = 0; i < length; i++) {
      const inputTensor = tf.tensor2d([input], [1, sequenceLength]);
      const preds = model.predict(inputTensor);
      const nextIndex = sample(preds, temperature);
      const nextWord = reverseWordIndex[nextIndex];
  
      result += ' ' + nextWord;
      input = input.slice(1).concat(nextIndex);
    }
  
    return result;
  }
  
  generateText("Ein Beispieltext").then(text => console.log(text));