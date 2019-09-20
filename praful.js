console.log('praful.js loaded')

urlParams = {}
window.location.search.slice(1).split('&').forEach(param => {
  const [key, value] = param.split('=')
  urlParams[key] = value
})

datasets = {
  'iris': {
    'url': "https://episphere.github.io/ai/data/iris.json",
    'labelName': "species"
  }
}

praful = async () => {
  const dataset = urlParams.dataset || "iris"
  const trainTestRatio = urlParams.split || 0.2
  const arch = (urlParams.arch && eval(urlParams.arch).length >= 3) ? eval(decodeURIComponent(urlParams.arch)) : [5, 3, 3]
  const activation = urlParams.activation || 'relu'
  const useBias = (urlParams.bias && eval(urlParams.bias)) || false
  const optimizer = urlParams.optimizer || 'sgd'
  const loss = urlParams.lossFn || 'categoricalCrossentropy'
  const metrics = (urlParams.metrics && eval(decodeURIComponent(urlParams.metrics)).length >= 1) ? eval(decodeURIComponent(urlParams.metrics)) : ['accuracy', 'precision']
  const epochs = (urlParams.epochs && parseInt(urlParams.epochs) != NaN) ? parseInt(urlParams.epochs) : 100
  const batchSize = (urlParams.batchSize && parseInt(urlParams.batchSize) != NaN) ? parseInt(urlParams.batchSize) : 8


  let [trainX, trainY, testX, testY] = await praful.getData(dataset, trainTestRatio)
  
  praful.data["training"] = {
    data: trainX,
    labels: trainY
  }
  praful.data["test"] = {
    data: testX,
    labels: testY
  }
  
  const modelConfig = {
    inputShape: trainX.shape[1],
    arch,
    activation,
    useBias,
    optimizer,
    loss,
    metrics
  }
  praful.model = praful.buildModel(modelConfig)

  const trainingConfig = {
    model: praful.model,
    epochs,
    batchSize,
    ...praful.data.training
  }
  await praful.trainModel(trainingConfig)
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
    // Convert n unique labels into an nxn tensor, with value set to 1
    // at the index corresponding to the actual label of the row, and 0
    // otherwise for each row. For instance, if (for iris) 'species' is 
    // 'setosa', the tensor will be [0 1 0] (because 'setosa' has index 1
    // in the `uniqueValues` Set).
    const distinctLabels = [...new Set(arr)]
    return arr.map(value => {
      const rowLabelsTensor = [0, 0, 0]
      rowLabelsTensor[distinctLabels.indexOf(value)] = 1
      return rowLabelsTensor
    })
  }
}

praful.getData = async (datasetName, trainTestRatio) => {
  const data = await praful.utils.request(datasets[datasetName].url)
  let [unlabeledData, labels] = await praful.preprocess(data, datasets[datasetName].labelName)
  const [testData, trainingData] = [unlabeledData.slice(0, data.length * trainTestRatio), unlabeledData.slice(data.length * trainTestRatio)]
  const [testLabels, trainingLabels] = [labels.slice(0, data.length * trainTestRatio), labels.slice(data.length * trainTestRatio)]
  return tf.tidy(() => [tf.tensor(trainingData), tf.tensor(trainingLabels), tf.tensor(testData), tf.tensor(testLabels)] )
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
  let { inputShape, arch, activation, useBias, optimizer, loss, metrics } = modelConfig
  inputShape = inputShape || praful.data.training.trainX.shape[1]
  arch = arch || [5, 3, 3]
  activation = activation || 'relu'
  useBias = typeof(useBias) !== 'undefined' ? useBias : false
  optimizer = optimizer || 'sgd'
  loss = loss || 'categoricalCrossentropy'
  metrics = metrics || ['accuracy', 'precision']

  let model = {}
  
  tf.tidy(() => {
    const layers = arch.map((layer, index) => {
      let layerArch = { 'units': layer, activation, useBias }
      if (index === 0) {
        layerArch.inputShape = inputShape
      } else if (index === arch.length - 1) {
        layerArch.activation = 'softmax'
      }
      return tf.layers.dense(layerArch)
    })
    model = tf.sequential({
      layers
    })
    console.log("Model Built! Architecture Summary:")
    model.summary()
    model.compile({ optimizer, loss, metrics })
  })
  return model
}

praful.trainModel = (trainingConfig) => {
  let { model, data, labels, epochs, batchSize, callbacks } = trainingConfig
  if (!model) {
    console.error("NO MODEL FOUND!")
  }
  epochs = epochs || 100
  batchSize = batchSize || 8
  callbacks = callbacks || {
    onTrainBegin: () => console.log("TRAINING BEGINS!"),
    onTrainEnd: () => console.log("TRAINING DONE!"),
    onEpochBegin: (epoch) => console.log(`Starting Epoch ${epoch}`),
    onEpochEnd: () => console.log("======================================================================================================"),
    onBatchEnd: (batch, logs) => console.log(`Accuracy for batch ${batch}: ${logs.acc}`)
  } 
  alert("Start Training?")
  model.fit(data, labels, {
    epochs,
    batchSize,
    callbacks
  }).then(info => {
    console.log("Model History: ", info.history)
    console.log("Final Accuracy: ", info.history.acc[epochs - 1])
  })
}