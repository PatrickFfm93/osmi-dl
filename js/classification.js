/**
 * classification.js
 *
 *
 * @version 0.1
 * @author  Patrick Bichiou, https://github.com/PatrickFfm93
 * @updated 2024-04-01
 *
 */

/* variable declarations */
const message = document.getElementById("message");
const imageSelect = document.getElementById("imageSelect");
const imageUploadButton = document.getElementById("imageUpload");
const imageDropArea = document.getElementById("imageDropArea");
const classifyButton = document.getElementById("classify");
const image = document.getElementById("img");
const classificationDiv = document.getElementById("classification");
const classifierOptions = {topk:6,};
const classifier = ml5.imageClassifier('MobileNet', classifierOptions, () => message.innerHTML = "Please upload or select an image!");

/* event listeners */
imageUpload.addEventListener("drop", (e) => {
  e.preventDefault();
  img.src = URL.createObjectURL(e.dataTransfer.items[0].getAsFile());
  imageDropArea.style.backgroundColor = "#fff";
});

imageUploadButton.addEventListener("change", (e) => {
  img.src = URL.createObjectURL(e.target.files[0]);
});

img.addEventListener("load", () => {
  message.innerHTML = "Image loaded successfully! Ready for classification...";
  classifyButton.style.display="block";
});

classifyButton.addEventListener("click", () => {
  message.innerHTML = "Classification in progress...";
  classifier.classify(img, classificationResultHandler);
});

const selectedImage = (imageName) => {
  switch(imageName){
    case "car":
      img.src = "assets/images/car.jpg";
      break;
    case "cat":
      img.src = "assets/images/cat.jpg";
      break;
    case "dog":
      img.src = "assets/images/dog.jpg";
      break;
    case "horse":
      img.src = "assets/images/horse.jpg";
      break;
    case "rocket":
      img.src = "assets/images/rocket.jpg";
      break;
    case "pear":
      img.src = "assets/images/pear.jpg";
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
