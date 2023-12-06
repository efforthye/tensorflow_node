const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { Image, createCanvas } = require('canvas');
const mobilenet = require('@tensorflow-models/mobilenet');

// 이미지 파일 형식 검사
const isSupportedImageFile = (fileName) => {
  const supportedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif'];
  return supportedExtensions.some((ext) => fileName.toLowerCase().endsWith(ext));
};

// 이미지를 로드하여 TensorFlow.js 텐서로 변환
const loadImage = async (path) => {
  const data = fs.readFileSync(path);
  const tensor = tf.node.decodeImage(data, 3);
  return tensor;
};

// 객체 탐지 및 네모박스 생성
const detectAndDrawBox = async (path, outputFilePath) => {
  console.log('하이2');
  try {
    const tensor = await loadImage(path);
    console.log({ tensor });
    try {
      console.log('모델 로딩 시도...');
      const model = await Promise.race([
        mobilenet.load(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('타임아웃 - 모델 로딩 실패')), 10000)),
      ]);
      console.log({ model });
    } catch (error) {
      console.error('오류 발생:', error);
    }

    const predictions = await model.classify(tensor);

    // 캔버스 초기화
    const [height, width] = tensor.shape.slice(0, 2);
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');

    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0);

        // 예측 결과 출력
        predictions.forEach((prediction, i) => {
          ctx.font = '20px Arial';
          ctx.fillStyle = 'red';
          ctx.fillText(`${prediction.className}: ${prediction.probability.toFixed(2)}`, 20, 20 * (i + 1));
        });

        resolve();
      };
      img.onerror = reject;
      img.src = tf.node.encodeJpeg(tensor);
    });

    // 이미지 저장
    const outputBuffer = canvas.toBuffer('image/png');
    fs.writeFileSync(outputFilePath, outputBuffer);

    tensor.dispose();
  } catch (error) {
    console.error('Error processing image:', path, error);
  }
};

// 모든 이미지 처리
const processAllImages = async () => {
  try {
    console.log('하이');
    const directoryPath = './images/raw';
    const files = fs.readdirSync(directoryPath);
    console.log({ files });

    for (let i = 0; i < files.length; i++) {
      console.log(isSupportedImageFile(files[i]));
      if (isSupportedImageFile(files[i])) {
        const filePath = `${directoryPath}/${files[i]}`;
        console.log({ filePath });
        const outputFilePath = `./images/output/mobilenet/mobilenet-${i + 1}.png`;
        console.log({ outputFilePath });
        await detectAndDrawBox(filePath, outputFilePath);
        console.log(`Processed ${files[i]} and saved as ${outputFilePath}`);
      }
    }
  } catch (error) {
    console.error('An error occurred during processing:', error);
  }
};

// 메인 함수 실행
(async () => {
  await processAllImages();
  console.log('All images processed.');
})();
