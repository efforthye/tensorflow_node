const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { Image, createCanvas } = require('canvas');
const cocoSsd = require('@tensorflow-models/coco-ssd');

// 이미지 파일 형식 검사
const isSupportedImageFile = (fileName) => {
  const supportedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif'];
  return supportedExtensions.some((ext) => fileName.toLowerCase().endsWith(ext));
};

// 이미지를 로드하여 캔버스에 그리기
const loadImage = async (path) => {
  const data = fs.readFileSync(path);
  const img = new Image();
  img.src = data;

  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  return { canvas, ctx };
};

// 고양이 탐지 및 네모박스 생성
const detectAndDrawBox = async (path, outputFilePath) => {
  const { canvas, ctx } = await loadImage(path);
  const model = await cocoSsd.load();

  const predictions = await model.detect(canvas);

  predictions.forEach((prediction) => {
    if (prediction.class === 'cat') {
      const [x, y, width, height] = prediction.bbox;
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      ctx.font = '20px Arial';
      ctx.fillStyle = 'red';
      ctx.fillText(prediction.class, x, y);
    }
  });

  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(outputFilePath, buffer);
};

// 모든 이미지 처리
const processAllImages = async () => {
  const directoryPath = './images/raw';
  const files = fs.readdirSync(directoryPath);

  for (let i = 0; i < files.length; i++) {
    if (isSupportedImageFile(files[i])) {
      const filePath = `${directoryPath}/${files[i]}`;
      const outputFilePath = `./images/output/cats/output-${i + 1}.png`;
      await detectAndDrawBox(filePath, outputFilePath);
      console.log(`Processed ${files[i]} and saved as ${outputFilePath}`);
    }
  }
};

processAllImages();
