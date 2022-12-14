let net;

const webCamElement = document.getElementById('webcam');
const webCamConfig = {
    facingMode: 'environment'
}

const positionFront = document.getElementById('positionFront');
const positionRear = document.getElementById('positionRear');

const classifier = knnClassifier.create();
const classes = ["Untrained","","",""];
const numberTrainings = ["","",""];

async function app(){

    net = await mobilenet.load();
    webcam = await tf.data.webcam(webCamElement,webCamConfig);

  while (true) {
    const img = await webcam.capture();

    const result = await net.classify(img);

    const activation = net.infer(img, 'conv_preds');
    var result2;
    try {
        result2 = await classifier.predictClass(activation);
    } catch (error) {
        result2 = {};
    }

    document.getElementById('className').innerText =  result[0].className;
    document.getElementById('probability').innerText =  Math.round(result[0].probability*100) + "%";

    try {
        document.getElementById('className2').innerText =  classes[result2.label];
        document.getElementById('probability2').innerText =  result2.confidences[result2.label];
    } catch (error) {
        document.getElementById('className2').innerText =  "Untrained";
        document.getElementById('probability2').innerText =  "Untrained";
    }
    // Dispose the tensor to release the memory.
    img.dispose();
    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

var count1 = 0;
var count2 = 0;
var count3 = 0;
async function addExample (classId) {
    const img = await webcam.capture();
    const activation = net.infer(img, true);

    classifier.addExample(activation, classId);

    classes[classId]=document.getElementById("clase-" + classId).value;

    switch (classId) {
        case 1:
            count1 = count1 + 1
            document.getElementById('numberOfTrainings-1').innerText = count1;
            break;
        case 2:
            count2 = count2 + 1
            document.getElementById('numberOfTrainings-2').innerText = count2;
            break;
        case 3:
            count3 = count3 + 1
            document.getElementById('numberOfTrainings-3').innerText = count3;
            break;
        default:
            document.getElementById('numberOfTrainings-1').innerText = "-";
            document.getElementById('numberOfTrainings-2').innerText = "-";
            document.getElementById('numberOfTrainings-3').innerText = "-";
    }
    
    //liberamos el tensor
    img.dispose()
}

positionFront.addEventListener('click', event => {
    webCamConfig.facingMode = 'user';
    webCamElement.setAttribute("width",224);
    webCamElement.setAttribute("height",224);
    let menu = document.getElementById('selectorPosition');
    while (menu.firstChild) {
        menu.removeChild(menu.firstChild);
    }
    app();
});

positionRear.addEventListener('click', event => {
    webCamConfig.facingMode = 'environment';
    webCamElement.setAttribute("width",224);
    webCamElement.setAttribute("height",224);
    let menu = document.getElementById('selectorPosition');
    while (menu.firstChild) {
        menu.removeChild(menu.firstChild);
    }
    app();
});