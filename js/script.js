const input = document.getElementById('textInput');
const bestPredictions = document.getElementById('bestPredictions');
const buttons = bestPredictions.children;
var text;

buttons.forEach(button => {
    button.style.display ='none';
    button.addEventListener('click', event =>{
        text = input.value.trim();
        text = text.concat(' ' + button.innerHTML);

        const words = text.toLowerCase().split(" ");
        input.value = text;
        input.focus();
        NextWord(words);
    });
});

const WORD_COUNT = 5;

const tokenizer_index = getJsonData('./../model/tokenizer_index_word.json');
const tokenizer_word = getJsonData('./../model/tokenizer_word_index.json');

var model;
(async function () {
    model = await tf.loadLayersModel("./../model/model.json");
})();

function NextWord(words){
    const wordArray = words.slice(-WORD_COUNT);
    const encoded = tokenizerEncode(tokenizer_word, wordArray);
    const tensorEncoded = tf.tensor(encoded, [1,5]);
    const pred = model.predict(tensorEncoded);
    buttons.forEach(button => {
        button.style.display = 'inline';
    });
    (async function () {
        let data = await pred.data();
        let temp = {};
        for(var i = 0; i < data.length;i++){
            temp[i] = data[i];
        }
        let mostProb = getLargestOfDict(temp, 5);
        let mostProbString = [];
        for(var i = 0; i< mostProb.length; i++){
            mostProbString[i] = tokenizer_index[parseInt(mostProb[i])];
            buttons[i].innerHTML = mostProbString[i];
        }
    })();
            
            
            
}

input.addEventListener('keydown', event => {
    text = input.value;
    const words = text.toLowerCase().split(" ");
    if (words.length >= WORD_COUNT){
        if(event.keyCode === 32 ){
            NextWord(words);
        }
    }else{
        buttons.forEach(button => {
            button.style.display ='none';
        });
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