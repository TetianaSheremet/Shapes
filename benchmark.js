const MODEL_PATH = './model/shape-model.json';
const IMAGES = [
  'test_images/256.png',
  'test_images/512.png',
  'test_images/1024.png',
  'test_images/2048.png',
  'test_images/4000.png'
];

const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const runBtn = document.getElementById('runBtn');

const PREPROCESS_REPEATS = 3;   
const INFER_REPEATS = 20;      
const WARMUP_RUNS = 3;         

async function loadModel() {
  statusEl.textContent = 'Завантаження моделі...';
  await tf.setBackend('webgl'); 
  await tf.ready();
  const model = await tf.loadLayersModel(MODEL_PATH);
  console.log('Бекенд:', tf.getBackend());
  statusEl.textContent = 'Модель готова';
  return model;
}

async function loadImage(url) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.decoding = 'async';
  img.src = url;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Не вдалося завантажити зображення: ${url}`));
  });
  return img;
}


async function measurePreprocessSec(img) {
  let totalMs = 0;
  for (let i = 0; i < PREPROCESS_REPEATS; i++) {
    const t0 = performance.now();
    const t = tf.tidy(() => {
      const px = tf.browser.fromPixels(img);
      const resized = tf.image.resizeBilinear(px, [64, 64]);
      return resized.toFloat().div(255).expandDims(0); 
    });
    
    await tf.nextFrame();
    const t1 = performance.now();
    totalMs += (t1 - t0);
    t.dispose();
  }
  return (totalMs / PREPROCESS_REPEATS) / 1000; // 
}


async function measureInferenceSec(model, img) {

  const input = tf.tidy(() => {
    const px = tf.browser.fromPixels(img);
    const resized = tf.image.resizeBilinear(px, [64, 64]);
    return resized.toFloat().div(255).expandDims(0);
  });

 
  for (let i = 0; i < WARMUP_RUNS; i++) {
    const out = model.predict(input);
    await out.data(); 
    out.dispose();
    await tf.nextFrame();
  }

  
  const start = performance.now();
  for (let i = 0; i < INFER_REPEATS; i++) {
    const out = model.predict(input);
    await out.data();
    out.dispose();
  }
  const end = performance.now();

  input.dispose();
  await tf.nextFrame();

  const avgSec = ((end - start) / INFER_REPEATS) / 1000;
  return avgSec;
}

async function runBenchmark() {
  try {
    runBtn.disabled = true;
    statusEl.textContent = 'Запуск тестування...';
    resultsEl.innerHTML = '';

    const table = document.createElement('table');
    table.innerHTML = `
      <tr>
        <th>Зображення</th>
        <th>Preprocess (сек)</th>
        <th>Inference (сек)</th>
        <th>Total≈ (сек)</th>
      </tr>`;
    resultsEl.appendChild(table);

    const model = await loadModel();

    
    const warmImg = await loadImage(IMAGES[0]);
    const warmInput = tf.tidy(() => {
      const px = tf.browser.fromPixels(warmImg);
      const resized = tf.image.resizeBilinear(px, [64, 64]);
      return resized.toFloat().div(255).expandDims(0);
    });
    for (let i = 0; i < WARMUP_RUNS; i++) {
      const out = model.predict(warmInput);
      await out.data();
      out.dispose();
      await tf.nextFrame();
    }
    warmInput.dispose();

    for (const path of IMAGES) {
      statusEl.textContent = `Обробка: ${path}`;
      const img = await loadImage(path);

      const preprocessSec = await measurePreprocessSec(img);
      const inferenceSec = await measureInferenceSec(model, img);
      const totalSec = (preprocessSec + inferenceSec);

      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${path}</td>
        <td>${preprocessSec.toFixed(3)}</td>
        <td>${inferenceSec.toFixed(3)}</td>
        <td>${totalSec.toFixed(3)}</td>`;
      table.appendChild(row);

      await tf.nextFrame();
    }

    tf.disposeVariables();
    statusEl.textContent = ' Тестування завершено ';
  } catch (e) {
    console.error(e);
    statusEl.textContent = ' Помилка під час тестування';
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener('click', runBenchmark);
