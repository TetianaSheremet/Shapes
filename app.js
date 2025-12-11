const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const saveBtn = document.getElementById("saveBtn");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const placeholder = document.getElementById("placeholder-text");
const progressContainer = document.querySelector(".progress-container");
const progressBar = document.getElementById("progressBar");
const result = document.getElementById("result");
const infoBtn = document.getElementById("infoBtn");
const infoBox = document.getElementById("infoBox");
const imgSize = document.getElementById("imgSize");
const imgResolution = document.getElementById("imgResolution");
const imgFormat = document.getElementById("imgFormat");

let uploadedImage = null;
let model = null;
const IMAGE_SIZE = 64;

// === Завантажуємо модель один раз ===
(async () => {
  try {
    model = await tf.loadLayersModel("model/shape-model.json");
    console.log(" Модель завантажено");
  } catch (err) {
    console.error("Не вдалося завантажити модель:", err);
  }
})();

// === Завантаження зображення ===
uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);


      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const x = (canvas.width / 2) - (img.width / 2) * scale;
      const y = (canvas.height / 2) - (img.height / 2) * scale;
      ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

      placeholder.style.display = "none";
      uploadedImage = img;

      
      imgSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
      imgResolution.textContent = `${img.width}×${img.height}`;
      imgFormat.textContent = file.type.split("/")[1]?.toUpperCase() || "N/A";
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});


clearBtn.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.style.display = "block";
  result.textContent = "Результат:";
  uploadedImage = null;
});


saveBtn.addEventListener("click", () => {
  const format = document.getElementById("formatSelect").value;

  if (format === "bmp") {
  
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    const headerSize = 54;
    const rowSize = Math.floor((24 * w + 31) / 32) * 4;
    const imageSize = rowSize * h;
    const fileSize = headerSize + imageSize;

    const buffer = new ArrayBuffer(fileSize);
    const view = new DataView(buffer);
    let p = 0;

   
    view.setUint16(p, 0x4D42, true); p += 2; // BM
    view.setUint32(p, fileSize, true); p += 4;
    p += 4; // reserved
    view.setUint32(p, headerSize, true); p += 4;
    view.setUint32(p, 40, true); p += 4;  // DIB header size
    view.setInt32(p, w, true); p += 4;
    view.setInt32(p, -h, true); p += 4;   // negative = top-down
    view.setUint16(p, 1, true); p += 2;   // planes
    view.setUint16(p, 24, true); p += 2;  // bits per pixel
    view.setUint32(p, 0, true); p += 4;   // compression
    view.setUint32(p, imageSize, true); p += 4;
    p += 16; // skip rest (0)

    
    let i = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const r = data[i++], g = data[i++], b = data[i++]; i++; // skip alpha
        view.setUint8(p++, b);
        view.setUint8(p++, g);
        view.setUint8(p++, r);
      }
    }
  // Створюємо файл і завантажуємо його
    const blob = new Blob([buffer], { type: "image/bmp" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "figure.bmp";
    link.click();
    URL.revokeObjectURL(link.href);
    return;
  }

   // --- Збереження у JPG або PNG ---
  const mimeType = format === "jpg" ? "image/jpeg" : "image/png";
  const link = document.createElement("a");
  link.download = `figure.${format}`;
  link.href = canvas.toDataURL(mimeType, 0.95);
  link.click();
});
// === 5. Кнопка "Інформація" ===
infoBtn.addEventListener("click", () => {
  infoBox.classList.toggle("hidden");
});

// === 6. Імітація анімації прогресу ===
function animateProgress(callback) {
  progressContainer.classList.remove("hidden");
  progressBar.style.width = "0%";
  let width = 0;
  const interval = setInterval(() => {
    width += 3;
    progressBar.style.width = width + "%";
    if (width >= 100) {
      clearInterval(interval);
      setTimeout(() => {
        progressContainer.classList.add("hidden");
        callback();
      }, 300);
    }
  }, 40);
}

// === 7. Підготовка зображення для моделі ===
function prepareImageForPrediction() {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(canvas).toFloat().div(255); // [H,W,3] 0..1
    let gray = t.mean(2);                                     // [H,W]   0..1

    const thr = gray.mean();                  // scalar
    const bw = gray.lessEqual(thr).toFloat(); // темніше за середнє -> 1 (фігура)

  
    const bw3 = tf.stack([bw, bw, bw], 2);   // [H,W,3]

   
    const resized = bw3.resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]);

    return resized.expandDims(0); // [1,64,64,3]
  });
}

// === 8. Кнопка "Розпізнати" ===
predictBtn.addEventListener("click", async () => {
  if (!uploadedImage) {
    alert("Будь ласка, завантажте фігуру спочатку!");
    return;
  }
  if (!model) {
    alert("Модель ще не завантажена. Зачекайте кілька секунд.");
    return;
  }

  result.textContent = "Розпізнавання...";
  animateProgress(async () => {
    const input = prepareImageForPrediction();
    const prediction = model.predict(input);
    const data = await prediction.data(); // Float32Array з ймовірностями

    
    let bestIdx = 0;
    for (let i = 1; i < data.length; i++) if (data[i] > data[bestIdx]) bestIdx = i;

    const classes = ["circle", "square", "triangle"]; // порядок як у тренуванні
    const shape = classes[bestIdx] || "невідомо";
    const confidence = (data[bestIdx] * 100).toFixed(1);

    result.textContent = `Результат: ${shape} (${confidence}%)`;
    input.dispose();
  });
});
