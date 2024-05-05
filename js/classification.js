/**
 * classification.js
 *
 *
 * @version 0.1
 * @author  Patrick Bichiou, https://github.com/PatrickFfm93
 * @updated 2024-05-05
 *
 */

/* variable declarations */
const message = document.getElementById("message");
const imageSelect = document.getElementById("imageSelect");
const imageUploadButton = document.getElementById("imageUpload");
const classifyButton = document.getElementById("classify");
const image = document.getElementById("img");
const classificationDiv = document.getElementById("classification");
const classifierOptions = {topk:6,};
const classifier = ml5.imageClassifier('MobileNet', classifierOptions, () => message.innerHTML = "Bitte wählen Sie ein Bild aus oder laden Sie eins hoch!");

/* event listeners */
imageUpload.addEventListener("drop", (e) => {
  e.preventDefault();
  if(e.dataTransfer.items[0].getAsFile().type == 'image') {
    img.src = URL.createObjectURL(e.dataTransfer.items[0].getAsFile());
  } else {
    alert("Falscher Dateityp: Es sind nur Bilder erlaubt");
  }
  
});

imageUploadButton.addEventListener("change", (e) => {
  if(e.target.files[0].type == 'image') {
   img.src = URL.createObjectURL(e.target.files[0]);
  } else {
    alert("Falscher Dateityp: Es sind nur Bilder erlaubt");
  }
});

img.addEventListener("load", () => {
  message.innerHTML = "Bilder erfolgreich geladen und bereit zur Klassifizierung...";
  classifyButton.style.display="block";
});

classifyButton.addEventListener("click", () => {
  message.innerHTML = "Klassifizierung läuft...";
  classifier.classify(img, classificationResultHandler);
});

const selectedImage = (imageName) => {
  switch(imageName){
    case "car":
      img.src = "assets/images/car.jpg";
      image.style.border = "1px solid green";
      break;
    case "cat":
      img.src = "assets/images/cat.jpg";
      image.style.border = "1px solid green";
      break;
    case "dog":
      img.src = "assets/images/dog.jpg";
      image.style.border = "1px solid green";
      break;
    case "horse":
      img.src = "assets/images/horse.jpg";
      image.style.border = "1px solid red";
      break;
    case "rocket":
      img.src = "assets/images/rocket.jpg";
      image.style.border = "1px solid red";
      break;
    case "pear":
      img.src = "assets/images/pear.jpg";
      image.style.border = "1px solid red";
      break;
    default:
      break; // do nothing if no valid option is selected.
  }
};

/* classification result function */
const classificationResultHandler = (error, results) => {
  // Display error in the console
  if (error) {
    console.error(error);
  } else {
    console.log(results);
    // The results are in an array ordered by confidence.
    let confidence = (results[0].confidence).toFixed(4);
    console.log(confidence);
    message.innerHTML = `Result: ${results[0].label}, Confidence: ${confidence} (${confidence * 100}%)`;
    console.log(results);
    plot(results);
  }
}
