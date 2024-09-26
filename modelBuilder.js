import * as tf from "https://esm.sh/@tensorflow/tfjs"
// import * as tfvis from "https://esm.sh/@tensorflow/tfjs-vis@1.5.1"

const DEFAULTS = {
    architecture: [
        {
            'type': "dense",
            'units': 5,
            'kernelSize': undefined,
            'filters': undefined,
            'poolSize': undefined,
            'strides': undefined,
            'activation': undefined,
            'bias': undefined
        },
        {
            'type': "dense",
            'units': 3,
            'kernelSize': undefined,
            'filters': undefined,
            'poolSize': undefined,
            'strides': undefined,
            'activation': undefined,
            'bias': undefined
        },
        {
            'type': "dense",
            'units': 3,
            'kernelSize': undefined,
            'filters': undefined,
            'poolSize': undefined,
            'strides': undefined,
            'activation': undefined,
            'bias': undefined
        },
    ],
    activation: "relu",
    bias: true,
    outputActivation: "softmax",
    optimizer: "sgd",
    lossFunc: "categoricalCrossentropy",
    metrics: ["accuracy"],
    numEpochs: 50,
    batchSize: 4,
    callbacks: {
        "onEpochEnd": (epochNum, metrics) => {
            console.log(`Metrics for epoch ${epochNum}: ${JSON.stringify(metrics)}`)
        }
    }
    //tfvis.show.fitCallbacks({
    //     'name': "Training",
    //     'tab': "Training"
    // }, ["loss", "acc"], {
    //     'callbacks': ["onEpochEnd"]
    // })
}

const utils = {
    request: (url, opts) => fetch(url, opts).then(res => res.json()),

    shuffleArray: (arr) => {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]]
        }
        return arr
    },

    convertLabelsToTensor: (arr) => {
        // Convert n unique labels into an nxn one-hot encoded tensor.
        const distinctLabels = [...new Set(arr.slice().sort())]
        return arr.map(value => {
            const rowLabelsTensor = [0, 0, 0]
            rowLabelsTensor[distinctLabels.indexOf(value)] = 1
            return rowLabelsTensor
        })
    },

    calculateMetrics: (predictedLabels, actualLabels) => {
        const predictions = tf.argMax(predictedLabels, 1).dataSync()
        const groundTruth = tf.argMax(actualLabels, 1).dataSync()

        const numCorrectPredictions = predictions.reduce((correctPreds, prediction, index) => {
            if (prediction === groundTruth[index]) {
                correctPreds += 1
            }
            return correctPreds
        }, 0)
        // More metrics could be added here.

       return { numCorrectPredictions }
    }
};

export class TFJSModel {
    constructor({ inputShape, architecture=DEFAULTS.architecture, activation=DEFAULTS.activation, bias=DEFAULTS.bias, outputActivation=DEFAULTS.outputActivation, optimizer=DEFAULTS.optimizer, lossFunc=DEFAULTS.lossFunc, metrics=DEFAULTS.metrics, numEpochs=DEFAULTS.numEpochs, batchSize=DEFAULTS.batchSize, callbacks=DEFAULTS.callbacks }) {
        this.inputShape = inputShape
        this.architecture = architecture || DEFAULTS.architecture
        this.activation = activation || DEFAULTS.activation
        this.bias = bias || DEFAULTS.bias
        this.outputActivation = outputActivation || DEFAULTS.outputActivation
        this.optimizer = optimizer || DEFAULTS.optimizer
        this.lossFunc = lossFunc || DEFAULTS.lossFunc
        this.metrics = metrics || DEFAULTS.metrics
        this.numEpochs = numEpochs || DEFAULTS.numEpochs
        this.batchSize = batchSize || DEFAULTS.batchSize
        this.callbacks = callbacks || DEFAULTS.callbacks

        this.model = undefined
    }

    preprocess = async (data, labelName, trainTestRatio) => {
        const shuffledData = utils.shuffleArray(data)

        let unlabeledData = [],
            correspLabels = []

        for (let row of shuffledData) {
            const {
                [labelName]: label, ...features
            } = row
            unlabeledData.push(features)
            correspLabels.push(label)
        }

        correspLabels = utils.convertLabelsToTensor(correspLabels)
        unlabeledData = unlabeledData.map(row => Object.values(row))
        return [unlabeledData, correspLabels]
    }

    async loadData({ datasetURL, labelFieldName, trainTestRatio }) {
        const data = await utils.request(datasetURL)
        
        let [unlabeledData, labels] = await this.preprocess(data, labelFieldName)
        
        const [trainingData, testData] = [unlabeledData.slice(0, data.length * trainTestRatio), unlabeledData.slice(data.length * trainTestRatio)]
        const [trainingLabels, testLabels] = [labels.slice(0, data.length * trainTestRatio), labels.slice(data.length * trainTestRatio)]
        
        this.dataset = {
            'trainingData': tf.tensor(trainingData),
            'trainingLabels': tf.tensor(trainingLabels),
            'testData': tf.tensor(testData),
            'testLabels': tf.tensor(testLabels)
        }
    }

    async build() {
        if (!Array.isArray(this.inputShape) && isNaN(this.inputShape)) {
            this.inputShape = this.dataset?.trainingData.shape[1]
        }
        if (typeof(this.inputShape) === 'undefined') {
            console.error("The shape of the input data cannot be determined to build the model. Try reloading the dataset or specify the correct shape when calling the function.")
            return
        }
        this.model = await new Promise(resolve => {

            tf.tidy(() => {
                const layers = this.architecture.map((layer, index) => {
                    let layerArch = {
                        'units': layer.units,
                        'kernelSize': layer.kernelSize,
                        'filters': layer.filters,
                        'poolSize': layer.poolSize,
                        'strides': layer.strides,
                        'activation': layer.activation || this.activation,
                        'bias': layer.bias || this.bias
                    }

                    if (index === 0) {
                        layerArch.inputShape = this.inputShape
                    } else if (index === this.architecture.length - 1) {
                        layerArch.activation = this.outputActivation
                    }
                    return tf.layers[layer.type](layerArch)
                })

                let model = tf.sequential({
                    layers
                })
                resolve(model)

            })

        })
    }

    train() {
        if (typeof(this.dataset) === 'undefined') {
            console.error("No dataset loaded! Please load the data first before starting training.")
            return
        }
        if (typeof(this.model) === 'undefined') {
            console.error("No model to train! Please build the model first before starting training.")
            return
        }
        
        return new Promise(resolve => {
            this.model.compile({
                'optimizer': this.optimizer,
                'loss': this.lossFunc,
                'metrics': this.metrics
            })

            this.model.fit(this.dataset.trainingData, this.dataset.trainingLabels, {
                'epochs': this.numEpochs,
                'batchSize': this.batchSize,
                'callbacks': this.callbacks
            }).then((info) => {
                resolve(info)
            })
        })
    }

    test() {
        return new Promise(resolve => {
            tf.tidy(() => {
                const predictions = this.model.predict(this.dataset.testData, {
                  'batchSize': this.batchSize
                })
                const { numCorrectPredictions } = utils.calculateMetrics(predictions, this.dataset.testLabels)
                
                // Return the proportion of correct predictions.
                resolve(numCorrectPredictions/this.dataset.testData.shape[0])
              })
        })
    }
}