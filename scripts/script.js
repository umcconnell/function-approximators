// Config
const inputs = {
    equation: {
        el: document.getElementById('equation'),
        defaultVal: 'x^2',
        parse: v => math.compile(v)
    },
    hiddenLayers: {
        el: document.getElementById('hidden-layers'),
        defaultVal: '4, 4, 4',
        parse: v =>
            v
                .trim()
                .split(/\s*,\s*/)
                .map(i => parseInt(i))
    },
    learningRate: {
        el: document.getElementById('learning-rate'),
        defaultVal: '0.05',
        parse: parseFloat
    },
    errorThresh: {
        el: document.getElementById('error-thresh'),
        defaultVal: '0.001',
        parse: parseFloat
    },
    activation: {
        el: document.getElementById('activation-function'),
        defaultVal: 'tanh',
        parse: v => v
    }
};

const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
const mainColor = getComputedStyle(document.documentElement).getPropertyValue(
    '--main-color'
);

// Plot
function plot(pretrainedNet = false) {
    try {
        if (!inputs.equation.el.value) return;
        const expr = inputs.equation.parse(inputs.equation.el.value);

        const xSource = math.range(-20, 21, 0.5).toArray();
        const xInput = math.range(-10, 11, 0.5).toArray();

        const ySource = xSource.map(x => expr.evaluate({ x: x }));
        const yInput = xInput.map(x => expr.evaluate({ x: x }));

        const yPredict = predict(xSource, xInput, yInput, pretrainedNet);

        const trace1 = {
            x: xSource,
            y: ySource,
            name: 'Source',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ddd' }
        };

        const trace2 = {
            x: xInput,
            y: yInput,
            name: 'Train',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#888' }
        };

        const trace3 = {
            x: xSource,
            y: yPredict,
            name: 'Prediction',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: mainColor }
        };

        Plotly.newPlot(
            'plot',
            [trace1, trace2, trace3],
            {
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                autosize: true,
                height: window.innerHeight - 11 * rem,
                margin: { t: 20, r: 0, b: 0, l: 0 }
            },
            { responsive: true }
        );
    } catch (err) {
        console.error(err);
        alert(err);
    }
}

// Logic
function predict(xSource, xInput, yInput, pretrainedNet) {
    const net = new brain.NeuralNetwork({
        hiddenLayers: inputs.hiddenLayers.parse(inputs.hiddenLayers.el.value),
        learningRate: inputs.learningRate.parse(inputs.learningRate.el.value),
        errorThresh: inputs.errorThresh.parse(inputs.errorThresh.el.value),
        activation: inputs.activation.parse(inputs.activation.el.value)
    });

    const xMax = Math.max(...xInput);
    // Remove Infinitys, NaN and Imaginary Numbers (objects)
    const yMax = Math.max(...yInput.filter(i => Number.isFinite(i)));

    if (pretrainedNet) {
        console.info('Using pretrained network');
        net.fromJSON(pretrainedNet);
    } else {
        console.info('Training network on-the-fly');
        const trainData = xInput
            .map((x, i) => [x, yInput[i]])
            .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
            .map(([x, y]) => ({
                input: [x / xMax],
                output: [y / yMax]
            }));

        net.train(trainData);
    }

    return xSource.map(x => net.run([x / xMax])[0] * yMax);
}

// UI & DOM Interaction
const eqForm = document.getElementById('eq-form');
const settingsForm = document.getElementById('settings-form');

const settingsModal = new Modal('#settings-modal');
const openSettingsModal = document.getElementById('open-settings-modal');

const aboutModal = new Modal('#about-modal');
const openAboutModal = document.getElementById('open-about-modal');

openSettingsModal.addEventListener('click', () => settingsModal.open());
openAboutModal.addEventListener('click', () => aboutModal.open());

eqForm.onsubmit = evt => {
    evt.preventDefault();
    plot();
};

settingsForm.onsubmit = evt => {
    settingsModal.close();
    evt.preventDefault();
    plot();
};

if (
    Object.values(inputs).every(({ el, defaultVal }) => el.value == defaultVal)
) {
    fetch('data/pretrainedNet.json')
        .then(data => data.json())
        .then(plot);
} else {
    plot();
}
