const IMAGE_SIZE = 64;
const CLASSES = ['circle', 'square', 'triangle'];

const folderInput = document.getElementById('folder');
const startBtn = document.getElementById('startBtn');
const bar = document.getElementById('bar');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const ctxChart = document.getElementById('chart').getContext('2d');

const chart = new Chart(ctxChart, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Accuracy', data: [], borderColor: '#22c55e', fill: false },
      { label: 'Val Accuracy', data: [], borderColor: '#3b82f6', fill: false },
      { label: 'Loss', data: [], borderColor: '#f59e0b', fill: false },
      { label: 'Val Loss', data: [], borderColor: '#ef4444', fill: false },
    ],
  },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Епоха' } },
      y: { min: 0, max: 1, title: { display: true, text: 'Значення' } },
    },
  },
});

function updateChart(epoch, logs) {
  chart.data.labels.push(epoch + 1);
  chart.data.datasets[0].data.push(logs.acc);
  chart.data.datasets[1].data.push(logs.val_acc);
  chart.data.datasets[2].data.push(logs.loss);
  chart.data.datasets[3].data.push(logs.val_loss);
  chart.update();
}

let filesByClass = { circle: [], square: [], triangle: [] };

folderInput.addEventListener('change', () => {
  filesByClass = { circle: [], square: [], triangle: [] };
  const files = Array.from(folderInput.files);
  for (const f of files) {
    const p = (f.webkitRelativePath || f.name).toLowerCase();
    if (p.includes('/circle/')) filesByClass.circle.push(f);
    else if (p.includes('/square/')) filesByClass.square.push(f);
    else if (p.includes('/triangle/')) filesByClass.triangle.push(f);
  }
  const msg = CLASSES.map(c => `${c}: ${filesByClass[c].length}`).join(' | ');
  log(`Знайдено файлів → ${msg}`);
  startBtn.disabled = !CLASSES.every(c => filesByClass[c].length > 0);
});

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  chart.data.labels = [];
  chart.data.datasets.forEach(d => (d.data = []));
  chart.update();
  try {
    await trainModel();
  } catch (e) {
    log('Помилка: ' + e.message);
    console.error(e);
  } finally {
    startBtn.disabled = false;
  }
});

function log(msg) {
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

async function readAsImageBitmap(file) {
  const url = URL.createObjectURL(file);
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  URL.revokeObjectURL(url);
  return bmp;
}

function toTensor(bmp) {
  return tf.tidy(() =>
    tf.browser.fromPixels(bmp)
      .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(255)
  );
}

function augment(img) {
  return tf.tidy(() => {
    const [h, w] = img.shape.slice(0, 2);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    const data = img.mul(255).toInt().dataSync();
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
      rgba[j] = data[i];
      rgba[j + 1] = data[i + 1];
      rgba[j + 2] = data[i + 2];
      rgba[j + 3] = 255;
    }
    ctx.putImageData(new ImageData(rgba, w, h), 0, 0);

    const angle = (Math.random() * 20 - 10) * Math.PI / 180;
    const scale = 0.85 + Math.random() * 0.3;
    const dx = Math.random() * 10 - 5;
    const dy = Math.random() * 10 - 5;

    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const c2 = off.getContext('2d');
    c2.translate(w / 2 + dx, h / 2 + dy);
    c2.rotate(angle);
    c2.scale(scale, scale);
    c2.drawImage(canvas, -w / 2, -h / 2);

    const imgData = c2.getImageData(0, 0, w, h);
    const d = imgData.data;
    const brightness = (Math.random() - 0.5) * 30;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.min(255, Math.max(0, d[i] + brightness));
      d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + brightness));
      d[i + 2] = Math.min(255, Math.max(0, d[i + 2] + brightness));
    }
    c2.putImageData(imgData, 0, 0);

    return tf.browser.fromPixels(off).toFloat().div(255);
  });
}

async function buildDataset(limitPerClass = 701, doAugment = true) {
  const xs = [], ys = [];
  for (let i = 0; i < CLASSES.length; i++) {
    const cls = CLASSES[i];
    const list = filesByClass[cls];
    const N = Math.min(limitPerClass, list.length);
    for (let j = 0; j < N; j++) {
      const bmp = await readAsImageBitmap(list[j]);
      let tensor = toTensor(bmp);
      if (doAugment) tensor = augment(tensor);
      xs.push(tensor);
      ys.push(i);
      if (xs.length % 100 === 0) {
        statusEl.textContent = `Завантажено зображень: ${xs.length}`;
        await tf.nextFrame();
      }
    }
  }
  const X = tf.stack(xs);
  const Y = tf.oneHot(tf.tensor1d(ys, 'int32'), CLASSES.length);
  xs.forEach(t => t.dispose());
  return { X, Y };
}

function makeModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  m.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.dropout({ rate: 0.25 }));

  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.4 }));
  m.add(tf.layers.dense({ units: CLASSES.length, activation: 'softmax' }));

  m.compile({
    optimizer: tf.train.adam(1e-3),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return m;
}

async function trainModel() {
  statusEl.textContent = 'Готуємо датасет...';
  const { X, Y } = await buildDataset(701, true);
  log(`Завантажено ${X.shape[0]} зразків`);

  const N = X.shape[0];
  const trainN = Math.floor(N * 0.8);
  const valN = N - trainN;

  const trainX = X.slice([0, 0, 0, 0], [trainN, IMAGE_SIZE, IMAGE_SIZE, 3]);
  const valX = X.slice([trainN, 0, 0, 0], [valN, IMAGE_SIZE, IMAGE_SIZE, 3]);
  const trainY = Y.slice([0, 0], [trainN, CLASSES.length]);
  const valY = Y.slice([trainN, 0], [valN, CLASSES.length]);

  const model = makeModel();
  log('Починаємо тренування...');
  statusEl.textContent = 'Тренування...';

  const EPOCHS = 20;
  await model.fit(trainX, trainY, {
    epochs: EPOCHS,
    batchSize: 32,
    validationData: [valX, valY],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const pct = Math.round(((epoch + 1) / EPOCHS) * 100);
        bar.value = pct;
        statusEl.textContent =
          `Епоха ${epoch + 1}/${EPOCHS} — loss ${logs.loss.toFixed(3)} | acc ${(logs.acc * 100).toFixed(1)}% | val_acc ${(logs.val_acc * 100).toFixed(1)}%`;
        log(statusEl.textContent);
        updateChart(epoch, logs);
        await tf.nextFrame();
      },
    },
  });

  statusEl.textContent = 'Зберігаємо модель...';
  await model.save('downloads://shape-model');
  log('Збережено: shape-model.json + shape-model.weights.bin');
  statusEl.textContent = 'Готово! Перемісти файли у /model/';
  X.dispose(); Y.dispose(); trainX.dispose(); trainY.dispose(); valX.dispose(); valY.dispose();
}



