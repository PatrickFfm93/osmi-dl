/**
 * plot.js
 *
 *
 * @version 0.1
 * @author  Patrick Bichiou, https://github.com/PatrickFfm93
 * @updated 2024-04-01
 *
 */

const plot = (plotData) => {
    const plotDataValues = [];
    const plotDataLabels = [];

    let confidenceSum = 0;

    for(let i = 0; i < plotData.length; i++){
        confidenceSum += plotData[i].confidence;
        plotDataValues.push(plotData[i].confidence);
        plotDataLabels.push(plotData[i].label);
    }

    if(confidenceSum < 1){
        plotDataValues.push(1 - confidenceSum);
        plotDataLabels.push("Others");
    }

    const dataPie = [{
        values: plotDataValues,
        labels: plotDataLabels,
        type: "pie",
    }];
    const dataBar = [{
        y: plotDataValues,
        x: plotDataLabels,
        type: "bar",
    }];
      
    let layout = {
        title: 'Prediction Results',
        font: {size: 14}
    };

    let config = {responsive: true}
      
    Plotly.newPlot('pieChart', dataPie, layout, config);
    Plotly.newPlot('barChart', dataBar, layout, config);

    document.getElementById("pieTab").click();
}

const changeChart = (evt, chartType) => {
    console.log(evt);
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(chartType).style.display = "block";
    if(evt !== null) evt.currentTarget.className += " active";
}