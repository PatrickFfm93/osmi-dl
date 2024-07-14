const input = document.getElementById('textInput');
// const bestPredictions = document.getElementById('bestPredictions');
// const buttons = bestPredictions.children;
var text;

// buttons.forEach(button => {
//     button.style.display ='none';
//     button.addEventListener('click', event =>{
//         text = input.value.trim();
//         text = text.concat(' ' + button.innerHTML);

//         const words = text.toLowerCase().split(" ");
//         input.value = text;
//         input.focus();
//         nextWord(words);
//     });
// });

const WORD_COUNT = 5;
const MAX_AUTO_PREDICTIONS = 10;
let autoMode = false;
let autoModeCount = 0;

const tokenizer_index = getJsonData('model/tokenizer_index_word.json');
const tokenizer_word = getJsonData('model/tokenizer_word_index.json');
let nextWordString = "";

var model;
(async function () {
    model = await tf.loadLayersModel("model/model.json");
})();

function nextWord(words){
    const wordArray = words.slice(-WORD_COUNT);
    const encoded = tokenizerEncode(tokenizer_word, wordArray);
    const tensorEncoded = tf.tensor(encoded, [1, 5]);
    const pred = model.predict(tensorEncoded);
    (async function () {
        let data = await pred.data();
        let temp = {};
        for(var i = 0; i < data.length;i++){
            temp[i] = data[i];
        }
        console.log(temp);
        let mostProb = getLargestOfDict(temp, 5); // # in auto mode this could be 10
        let mostProbString = [];
        console.log(mostProb);
        document.getElementById('mostProbWords').innerHTML = "";
        for(var i = 0; i< mostProb.length; i++){
            mostProbString[i] = tokenizer_index[parseInt(mostProb[i])];
            document.getElementById('mostProbWords').innerHTML += `${mostProbString[i]} (${(mostProb[i][1] * 100).toFixed(2)}%) `;
        }
        nextWordString = tokenizer_index[parseInt(mostProb[0])];
        document.getElementById('next').disabled = false;
        if(autoMode && autoModeCount < MAX_AUTO_PREDICTIONS){
            autoModeCount++;
            input.value = input.value.concat(' ' + nextWordString);
            text = input.value;
            const words = text.toLowerCase().split(" ");
            setTimeout(() => nextWord(words), 1000);
        } else if (autoMode && autoModeCount === MAX_AUTO_PREDICTIONS){
            autoModeCount = 0;
            autoMode = false;
            input.value = input.value.concat(' ' + nextWordString);
            document.getElementById('auto').innerHTML = "Auto";
        }
    })();  
}

input.addEventListener('keydown', event => {
    text = input.value;
    const words = text.toLowerCase().split(" ");
    if (words.length >= WORD_COUNT && event.keyCode === 32){
            document.getElementById('prediction').disabled = false;
            document.getElementById('auto').disabled = false;
    } else {
        document.getElementById('prediction').disabled = true;
        document.getElementById('auto').disabled = true;
    }
});

document.getElementById('prediction').addEventListener('click', event => {
    text = input.value;
    const words = text.toLowerCase().split(" ");
    nextWord(words);
});

document.getElementById('next').addEventListener('click', event => {
    input.value = input.value.concat(' ' + nextWordString);
    text = input.value;
    const words = text.toLowerCase().split(" ");
    nextWord(words);
});

document.getElementById('reset').addEventListener('click', async event => {
    text = undefined;
    input.value = "";
    autoMode = false;
    autoModeCount = 0;  
    nextWordString = "";
    document.getElementById('prediction').disabled = true;
    document.getElementById('next').disabled = true;
    document.getElementById('mostProbWords').innerHTML += "";
    model = await tf.loadLayersModel("model/model.json")
});

document.getElementById('auto').addEventListener('click', event => {
    if(autoMode){
        autoMode = false;
        autoModeCount = 0;
        document.getElementById('auto').innerHTML = "Auto";
    } else {
        autoMode = true;
        document.getElementById('auto').innerHTML = "Stop";
        text = input.value;
        const words = text.toLowerCase().split(" ");
        autoModeCount++;
        nextWord(words);
    }
    
});

function getLargestOfDict(dict, count){
    let items = Object.keys(dict).map(function(key){
        return [key, dict[key]];
    });

    items.sort(function(first, second){
        return second[1] - first[1];
    });

    return items.slice(0, count);
}

function tokenizerEncode(tokenizer, wordArray){
    var encoded = [];
    wordArray.forEach(word => {
        encoded.push(tokenizer[word]);
    });
    return encoded;
}


function getJsonData(filePath) {
    var result = null;
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("GET", filePath, false);
    xmlhttp.send();
    if (xmlhttp.status==200) {
      result = xmlhttp.responseText;
    }
    return JSON.parse(result);
  }