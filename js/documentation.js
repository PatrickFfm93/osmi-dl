const summary = document.getElementById('modelsummary');


(async function () {
    learned_model = await tf.loadLayersModel("model/model.json");
    tfvis.show.modelSummary(
        summary, 
        learned_model
      );
  
   
})();

