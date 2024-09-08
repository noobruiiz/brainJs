
    const brain = require('brain.js');
const fs = require('fs');

const net = new brain.NeuralNetwork({
    hiddenLayers: [3]
});

const trainingData = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];

net.train(trainingData, {
    iterations: 20000,
    log: true,
    logPeriod: 1000
});

const saveModel = (filename) => {
    const json = net.toJSON();
    fs.writeFileSync(filename, JSON.stringify(json));
    console.log('Modelo guardado en', filename);
};

const loadModel = (filename) => {
    if (fs.existsSync(filename)) {
        const json = JSON.parse(fs.readFileSync(filename));
        net.fromJSON(json);
        console.log('Modelo cargado desde', filename);
    } else {
        console.log('No se encontró el archivo del modelo, entrenando desde cero.');
    }
};

saveModel('model.json');

loadModel('model.json');

const roundOutput = (output) => Math.round(output[0]);

const predictions = [
    { input: [0, 0], expected: [0] },
    { input: [0, 1], expected: [1] },
    { input: [1, 0], expected: [1] },
    { input: [1, 1], expected: [0] }
];

predictions.forEach(({ input, expected }) => {
    const output = net.run(input);
    console.log(`Entrada: ${input}, Predicción: ${roundOutput(output)}, Esperado: ${expected}`);
});
