import {layers, linspace, losses, Sequential, tensor2d, tidy, train, util} from '@tensorflow/tfjs'
import fetch from 'node-fetch'

// tensorflow推荐使用@tensorflow/tfjs-node包, 但是好像还有好多bug, 懒得搞了

const getData = async (): Promise<any[]> => {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  return carsData.map((car: any) => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  })).filter((car: any) => (car.mpg != null && car.horsepower != null));
}


/**
 * 将数据转换为tensor类型
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
const convertToTensor = (data: any[]) => {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tidy(() => {
    // Step 1. Shuffle the data
    util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

    const inputTensor = tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}


const app = async () => {
  console.log("加载数据");
  // Test app
  const data = await getData(); // x: d.horsepower   y: d.mpg

  console.log("定义模型");
  // Create a sequential model
  const model = new Sequential();
  // 输入层 Add a single input layer
  model.add(layers.dense({inputShape: [1], units: 1, useBias: true}));
  // 输出层 Add an output layer
  model.add(layers.dense({units: 1, useBias: true}));

  // 加载数据
  const {inputs, labels, inputMax, inputMin, labelMax, labelMin,} = convertToTensor(data);
  // 定义/配置模型
  model.compile({
    optimizer: train.adam(),
    loss: losses.meanSquaredError,
    metrics: ['mse'],
  });

  console.log("训练...");
  // 开始训练
  const batchSize = 32;
  const epochs = 50;
  await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true
  });


  const xs = linspace(0, 1, 100);
  const preds = model.predict(xs.reshape([100, 1]));

  console.log("预测结果:", preds);
}

app();