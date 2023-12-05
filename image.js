const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');
const { Image, createCanvas } = require('canvas');

// 이미지를 가져와 캔버스에 그린후 텐서로 변환
const loadImage = async (path) => {
  const data = fs.readFileSync(path);
  const img = new Image();
  img.src = data;

  // 원본 이미지 크기를 사용하는 대신 224x224로 설정
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');

  // 캔버스에 이미지를 그리기 전에 크기를 조정
  ctx.drawImage(img, 0, 0, 224, 224);

  const tensor = tf.browser.fromPixels(canvas);
  return tensor.div(255);
};

// 이미지를 가져와 텐서플로우 모델을 로드하여 이미지를 분류
const classifyImage = async (path) => {
  const image = await loadImage(path);
  const model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  );
  const predictions = model.predict(image);
  predictions.print();
};

// 이미지를 arraybuffer 형식으로 다운로드 후 이미지 객체 반환
const downloadImage = async (url) => {
  const response = await axios({
    url,
    responseType: 'arraybuffer',
  });
  const buf = Buffer.from(response.data, 'binary');
  const img = new Image();
  img.src = buf;
  return img;
};

// 예측할 이미지를 가져와 크기 조절 후 텐서플로우 모델을 통하여 예측
const loadImageAndPredict = async (path) => {
  const img = await loadImage(path);

  // 이미지 크기 조정
  const resized = tf.image.resizeBilinear(img, [224, 224]);

  // 이미지 정규화
  const offset = tf.scalar(127.5);
  const normalized = resized.sub(offset).div(offset);

  // 배치 차원 추가
  const batched = normalized.reshape([1, 224, 224, 3]);

  // 모델 로드 및 예측
  const model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  );
  const predictions = model.predict(batched);
  predictions.print();
};

// 캔버스 이미지를 파일 형식으로 저장
const saveCanvasToFile = () => {
  const width = 800;
  const height = 600;

  // 캔버스 생성
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // 캔버스에 직사각형을 그린다.
  ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
  ctx.fillRect(100, 100, 200, 200);

  // 캔버스의 내용을 이미지 파일로 저장
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync('./images/output/canvas-output.png', buffer);
};

const imageURL = './images/train/cats.png';
classifyImage(imageURL);
loadImageAndPredict(imageURL);
saveCanvasToFile();
