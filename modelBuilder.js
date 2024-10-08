import * as tf from "https://esm.sh/@tensorflow/tfjs"

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

    calculateMetrics: (predictedLabels, actualLabels) => {
        const predictions = tf.argMax(predictedLabels, 1).dataSync()
        const groundTruth = tf.argMax(actualLabels, 1).dataSync()

        const numCorrectPredictions = predictions.reduce((correctPreds, prediction, index) => {
            if (prediction === groundTruth[index]) {
                correctPreds += 1
            }
            return correctPreds
        }, 0)

        return { numCorrectPredictions }
    }
};

/**
 * Class representing a TensorFlow.js model.
 */
export class TFJSModel {
    /**
     * Creates an instance of the TFJSModel.
     * @param {Object} options - The model options.
     * @param {Array<Object>|Array<number>} [options.architecture=[5,3,3]] - The model architecture configuration. Could be an array of numbers corresponding to the number of units per densely connected layer of the neural network, or an object with more granular properties.
     * @param {string} [options.activation="relu"] - The activation function to be used in all the hidden layers. Could be one of: 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'.
     * @param {boolean} [options.bias=true] - Whether to include a bias term in the layers.
     * @param {string} [options.outputActivation="softmax"] - The activation function for the output layer. Could be one of: 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'.
     * @param {string} [options.optimizer="sgd"] - The optimizer to use for training. Could be one of: 'sgd'|'momentum'|'adagrad'|'adadelta'|'adam'|'adamax'|'rmsprop' .
     * @param {string} [options.lossFunc="categoricalCrossentropy"] - The loss function to use for training. Could be one of: 'absoluteDifference'|'computeWeightedLoss'|'cosineDistance'|'hingeLoss'|'huberLoss'|'logLoss'|'meanSquaredError'|'sigmoidCrossEntropy'|'softmaxCrossEntropy'|'binaryAccuracy'|'binaryCrossentropy'|'categoricalAccuracy'|'categoricalCrossentropy'|'cosineProximity'|'meanAbsoluteError'|'meanAbsolutePercentageError'|'meanSquaredError'|'precision'|'recall'|'sparseCategoricalAccuracy'.
     * @param {Array<string>} [options.metrics=["accuracy"]] - The metrics to track during training. Could be one or more of: 'binaryAccuracy'|'binaryCrossentropy'|'categoricalAccuracy'|'categoricalCrossentropy'|'cosineProximity'|'meanAbsoluteError'|'meanAbsolutePercentageError'|'meanSquaredError'|'precision'|'r2Score'|'recall'|'sparseCategoricalAccuracy' .
     * @param {number} [options.numEpochs=50] - The number of iterations to run the training for.
     * @param {number} [options.batchSize=4] - The batch size to be used every epoch.
     * @param {Object} [options.callbacks] - List of callbacks to be executed at various points during training. Valid callbacks include: 'onTrainBegin(logs)'|'onTrainEnd(logs)'|'onEpochBegin(epoch, logs)'|'onEpochEnd(epoch, logs)'|'onBatchBegin(batch, logs)'|'onBatchEnd(batch, logs)'|'onYield(epoch, batch, logs)'. Includes the 'onEpochEnd' callback by default.
     */
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

    /**
     * Preprocesses the data by shuffling it and separating features from labels. Called by default on loading the data.
     * @param {Array<Object>} data - The input data.
     * @param {string} labelName - The name of the label field.
     * @returns {Promise<Array<Array<any>, Array<any>>>} The unlabeled data and corresponding labels.
     */
    preprocess (data, labelName) {
        const shuffledData = utils.shuffleArray(data)

        let unlabeledData = [],
            correspLabels = []

        for (let row of shuffledData) {
            const {
                [labelName]: label, ...features
            } = row
            unlabeledData.push(Object.values(features))
            correspLabels.push(label)
        }

        return [unlabeledData, correspLabels]
    }

    /**
     * Loads data from a specified URL and preprocesses it for training/testing.
     * @param {Object} params - Parameters for loading data.
     * @param {string} params.datasetURL - The URL to the dataset file.
     * @param {string} params.labelFieldName - The name of the label field.
     * @param {number} [params.trainTestRatio=0.8] - The ratio for splitting data into training and testing sets.
     */
    async loadData({ datasetURL, labelFieldName, trainTestRatio=0.8 }) {
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

    /**
     * Compiles the model architecture based on the provided input shape and architecture configuration.
     */
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
                        'type': layer?.type || "dense",
                        'units': layer?.units || layer,
                        'kernelSize': layer?.kernelSize,
                        'filters': layer?.filters,
                        'poolSize': layer?.poolSize,
                        'strides': layer?.strides,
                        'activation': layer?.activation || this.activation,
                        'bias': layer?.bias || this.bias
                    }

                    if (index === 0) {
                        layerArch.inputShape = this.inputShape
                    } else if (index === this.architecture.length - 1) {
                        layerArch.activation = this.outputActivation
                    }
                    return tf.layers[layerArch.type](layerArch)
                })

                let model = tf.sequential({
                    layers
                })
                resolve(model)

            })

        })
    }

    /**
     * Trains the model using the loaded dataset.
     * @returns {Promise<Object>} A promise that resolves to the training information, including the values of the final loss and the metrics specified.
     */
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

    /**
     * Validates the model on the test dataset and calculates the proportion of correct predictions.
     * @returns {Promise<number>} A promise that resolves to the proportion of correct predictions (accuracy for now, more metrics could be added later).
     */
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