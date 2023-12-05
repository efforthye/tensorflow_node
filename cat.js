const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { Image, createCanvas } = require('canvas');
const cocoSsd = require('@tensorflow-models/coco-ssd');

// 이미지를 가져와 캔버스에 그린다.
const loadImage = async (path) => {
  const data = fs.readFileSync(path);
  const img = new Image();
  img.src = data;

  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  return { canvas, ctx };
};

// 고양이에 네모박스를 그린다.
const detectAndDrawBox = async (path) => {
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
  fs.writeFileSync('./images/output/cat/cat-detected.png', buffer);
};

const imagePath = './images/raw/cats2.jpg';
detectAndDrawBox(imagePath);
