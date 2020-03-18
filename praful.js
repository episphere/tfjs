console.log('praful.js loaded')
modelPath = window.location.host.includes('localhost') ? 'localstorage://iris-model' : 'downloads://iris-model'

urlParams = {}
if (!window.location.origin.includes("repl.it")) {
  window.location.search.slice(1).split('&').forEach(param => {
    const [key, value] = param.split('=')
    urlParams[key] = value
  })
}

// replParams = (qs) => {
//   qs.slice(1).split('&').forEach(param => {
//     const [key, value] = param.split('=')
//     urlParams[key] = value
//   })
// }

datasets = {
  'iris': {
    'url': "https://episphere.github.io/ai/data/iris.json",
    'labelName': "species"
  }
}

praful = async () => {
  praful.visor = tfvis.visor()
  const dataset = urlParams.dataset || "iris"
  const trainTestRatio = urlParams.split ? parseFloat(urlParams.split) : 0.2
  const arch = (urlParams.arch && eval(urlParams.arch).length >= 3) ? eval(decodeURIComponent(urlParams.arch)) : [5, 3, 3]
  const activation = urlParams.activation || "relu"
  const useBias = (urlParams.bias && eval(urlParams.bias)) || false
  const optimizer = urlParams.optimizer || "sgd"
  const loss = urlParams.lossFn || "categoricalCrossentropy"
  const metrics = (urlParams.metrics && eval(decodeURIComponent(urlParams.metrics)).length >= 1) ? eval(decodeURIComponent(urlParams.metrics)) : ["accuracy", "precision"]
  const epochs = (urlParams.epochs && parseInt(urlParams.epochs) != NaN) ? parseInt(urlParams.epochs) : 100
  const batchSize = (urlParams.batchSize && parseInt(urlParams.batchSize) != NaN) ? parseInt(urlParams.batchSize) : 8


  let [trainX, trainY, testX, testY] = await praful.getData(dataset, trainTestRatio)

  praful.data["training"] = {
    'data': trainX,
    'labels': trainY
  }
  praful.data["test"] = {
    'data': testX,
    'labels': testY
  }
  try {
    praful.model = await tf.loadLayersModel(modelPath)
  } catch (e) {
    console.error("Error Loading Model: ", e)
  }

  if (!praful.model || (praful.model && !confirm("Local Model Found. Press Ok to use it or Cancel to build new model."))) {
    const modelConfig = {
      'inputShape': trainX.shape[1],
      arch,
      activation,
      useBias,
    }
    praful.model = praful.buildModel(modelConfig)
  }
  
  praful.renderVisualizations(praful.model)
  console.log("Model Architecture: ")
  praful.model.summary()
  console.log("Weights: ")
  praful.model.weights.forEach(w => w.val.print())
  
  if (confirm("Train Model?")) {
    const trainingConfig = {
      'model': praful.model,
      epochs,
      batchSize,
      optimizer,
      loss,
      metrics,
      ...praful.data.training
    }
    await praful.trainModel(trainingConfig)
  }

  if (confirm("Test Model?")) {
    const testConfig = {
      model: praful.model,
      batchSize,
      ...praful.data.test
    }
    await praful.testModel(testConfig)
  }
}

praful.renderVisualizations = async (model) => {
  tfvis.show.modelSummary({
    'name': "Model Architecture",
    'tab': "Model"
  }, model)
  model.layers.forEach(async (layer, index) => {
    tfvis.show.layer({
      'name': `Layer ${index+1}`,
      'tab': "Model"
    }, layer)
  })
}

praful.data = {}

praful.utils = {
  request: (url, opts) => fetch(url, opts).then(res => res.json()),

  shuffleArray: (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
  },

  convertLabelsToTensor: (arr) => {
    /* Convert n unique labels into an nxn tensor, with value set to 1
     * at the index corresponding to the actual label of the row, and 0
     * otherwise for each row. For instance, if (for iris) 'species' is 
     * 'setosa', the tensor will be [0 1 0] (because 'setosa' has index 1
     * in the `uniqueValues` Set). */
    const distinctLabels = [...new Set(arr.slice().sort())] // <Array>.sort mutates the original array, so slicing to create a copy first and then sorting.
    return arr.map(value => {
      const rowLabelsTensor = [0, 0, 0]
      rowLabelsTensor[distinctLabels.indexOf(value)] = 1
      return rowLabelsTensor
    })
  },

  calculateMetrics: async (predictedLabels, actualLabels) => {
    const predictions = tf.argMax(predictedLabels, 1).dataSync()
    const groundTruth = tf.argMax(actualLabels, 1).dataSync()
    console.log("Predictions: ", predictions)
    console.log("Actual Labels: ", groundTruth)
    const numCorrectPredictions = predictions.reduce((correctPreds, prediction, index) => {
      // console.log(`Predicted vs Actual Labels for Test Observation ${index + 1} : ${[prediction, groundTruth[index]]}`)
      if (prediction === groundTruth[index]) {
        correctPreds += 1
      }
      return correctPreds
    }, 0)
    console.log(`Test Accuracy: ${100 * numCorrectPredictions/groundTruth.length}`)
    const distinctLabels = new Set(groundTruth)
    const confusionMatrix = await tfvis.metrics.confusionMatrix(tf.argMax(actualLabels, 1), tf.argMax(predictedLabels, 1), distinctLabels.size)
    tfvis.render.table({
      'name': "Confusion Matrix",
      'tab': "Evaluation"
    }, {
      headers: ['0', '1', '2'],
      values: confusionMatrix
    })
  }
}

praful.getData = async (datasetName, trainTestRatio) => {
  const data = await praful.utils.request(datasets[datasetName].url)
  let [unlabeledData, labels] = await praful.preprocess(data, datasets[datasetName].labelName)
  const [testData, trainingData] = [unlabeledData.slice(0, data.length * trainTestRatio), unlabeledData.slice(data.length * trainTestRatio)]
  const [testLabels, trainingLabels] = [labels.slice(0, data.length * trainTestRatio), labels.slice(data.length * trainTestRatio)]
  return tf.tidy(() => [tf.tensor(trainingData), tf.tensor(trainingLabels), tf.tensor(testData), tf.tensor(testLabels)])
}

praful.preprocess = async (data, labelName, trainTestRatio) => {
  const shuffledData = praful.utils.shuffleArray(data)

  // Separate the label from the features.
  let unlabeledData = [],
    correspLabels = []

  for (let row of shuffledData) {
    const {
      [labelName]: label, ...features
    } = row
    unlabeledData.push(features)
    correspLabels.push(label)
  }

  // Assumption made for Iris: Features are all numeric values.
  correspLabels = praful.utils.convertLabelsToTensor(correspLabels)
  unlabeledData = unlabeledData.map(row => Object.values(row))
  return [unlabeledData, correspLabels]
}

praful.buildModel = (modelConfig) => {
  let {
    inputShape,
    arch,
    activation,
    useBias,
  } = modelConfig
  inputShape = inputShape || praful.data.training.trainX.shape[1]
  arch = arch || [5, 3, 3]
  activation = activation || "relu"
  useBias = typeof (useBias) !== "undefined" ? useBias : false

  let model = {}

  tf.tidy(() => {
    const layers = arch.map((layer, index) => {
      let layerArch = {
        'units': layer,
        activation,
        useBias
      }
      if (index === 0) {
        layerArch.inputShape = inputShape
      } else if (index === arch.length - 1) {
        layerArch.activation = "softmax"
      }
      return tf.layers.dense(layerArch)
    })
    model = tf.sequential({
      layers
    })
  })
  return model
}

praful.trainModel = (trainingConfig) => {
  let {
    model,
    data,
    labels,
    epochs,
    batchSize,
    optimizer,
    loss,
    metrics,
    callbacks
  } = trainingConfig
  
  if (!model) {
    console.error("NO MODEL FOUND!")
  }
  
  epochs = epochs || 100
  batchSize = batchSize || 8
  optimizer = optimizer || "sgd"
  loss = loss || "categoricalCrossentropy"
  metrics = metrics || ["accuracy", "precision"]
  callbacks = callbacks || tfvis.show.fitCallbacks({
    'name': "Training",
    'tab': "Training"
  }, ["loss", "acc"], {
    'callbacks': ["onEpochEnd"]
  })
  
  // {
  //   onTrainBegin: () => {
  //     console.log("TRAINING BEGINS!")
  //   },
  //   onEpochBegin: (epoch) => console.log(`=================================\nStarting Epoch ${epoch+1}`),
  //   onEpochEnd: (epoch, logs) => console.log(`Accuracy for epoch ${epoch+1}: ${logs.acc}`),
  //   onTrainEnd: () => {
  //     console.log("TRAINING DONE!")
  //     
  //   }
  // }
  
  model.compile({ optimizer, loss, metrics })
  return model.fit(data, labels, {
    epochs,
    batchSize,
    callbacks
  }).then(async (info) => {
    await model.save(modelPath)
    console.log("Model History: ", info.history)
    console.log("Final Accuracy: ", info.history.acc[epochs - 1])
    console.log("Final Weights: ")
    model.weights.forEach(layerWeights => {
      console.log(`Layer ${layerWeights.name} has weights of shape ${layerWeights.shape}. Values:`)
      layerWeights.val.print()
    })
  })
}

praful.testModel = async (testConfig) => {
  console.log("Starting Test!")
  tf.tidy(() => {
    let {
      model,
      data,
      labels,
      batchSize
    } = testConfig
    if (!model) {
      console.error("NO MODEL FOUND!")
    }
    const predictions = model.predict(data, {
      batchSize
    })
    
    praful.utils.calculateMetrics(predictions, labels)
    
  })
}

praful.loadDataset = async () => {
  const urlToCSV = urlParams.csvPath
  const csvConfig = null
  if (urlParams.labelCol) {
    csvConfig = {
      columnConfigs: {}
    }
    const labelColumn = urlParams.labelCol
    csvConfig.columnConfigs[labelColumn] = {
      isLabel: true
    }
  }
  const dataset = tf.data.csv(urlToCSV, csvConfig)
  console.log(await dataset.columnNames())
  dataset.forEachAsync((row) => console.log(row ))
}