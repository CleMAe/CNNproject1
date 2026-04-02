const healthIndicator = document.getElementById("health-indicator");
const modelName = document.getElementById("model-name");
const deviceName = document.getElementById("device-name");
const imageSize = document.getElementById("image-size");

const dropZone = document.getElementById("drop-zone");
const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const fileMeta = document.getElementById("file-meta");
const requestHint = document.getElementById("request-hint");

const predictButton = document.getElementById("predict-button");
const resetButton = document.getElementById("reset-button");
const copyJsonButton = document.getElementById("copy-json-button");

const predictedLabel = document.getElementById("predicted-label");
const confidenceText = document.getElementById("confidence-text");
const elapsedMs = document.getElementById("elapsed-ms");
const imageDimensions = document.getElementById("image-dimensions");
const jsonOutput = document.getElementById("json-output");

const probOkValue = document.getElementById("prob-ok-value");
const probDefectValue = document.getElementById("prob-defect-value");
const probOkBar = document.getElementById("prob-ok-bar");
const probDefectBar = document.getElementById("prob-defect-bar");

let selectedFile = null;

function setHealthState(payload, isError = false) {
  if (isError) {
    healthIndicator.textContent = "不可用";
    healthIndicator.className = "status-pill error";
    modelName.textContent = "服务连接失败";
    deviceName.textContent = "-";
    imageSize.textContent = "-";
    return;
  }

  healthIndicator.textContent = payload.status === "ok" ? "运行正常" : "启动中";
  healthIndicator.className = payload.status === "ok" ? "status-pill ok" : "status-pill pending";
  modelName.textContent = `模型：${payload.model_name}`;
  deviceName.textContent = payload.device;
  imageSize.textContent = `${payload.image_size} px`;
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    setHealthState(payload, !response.ok);
  } catch (error) {
    setHealthState({}, true);
  }
}

function resetResults() {
  predictedLabel.textContent = "等待检测";
  confidenceText.textContent = "置信度：-";
  elapsedMs.textContent = "-";
  imageDimensions.textContent = "-";
  probOkValue.textContent = "-";
  probDefectValue.textContent = "-";
  probOkBar.style.width = "0%";
  probDefectBar.style.width = "0%";
  jsonOutput.textContent = '{\n  "message": "等待接口返回结果"\n}';
}

function applyPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewImage.style.display = "block";
    previewPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);

  fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  requestHint.textContent = "图片已就绪，可以开始检测。";
  predictButton.disabled = false;
}

function clearSelection() {
  selectedFile = null;
  imageInput.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  previewPlaceholder.style.display = "grid";
  fileMeta.textContent = "尚未选择文件";
  requestHint.textContent = "请先上传图片。";
  predictButton.disabled = true;
  resetResults();
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    requestHint.textContent = "请选择图片文件。";
    return;
  }
  selectedFile = file;
  applyPreview(file);
  resetResults();
}

async function predictImage() {
  if (!selectedFile) {
    requestHint.textContent = "请先上传图片。";
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  predictButton.textContent = "检测中...";
  requestHint.textContent = "正在调用 /predict，请稍候。";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    jsonOutput.textContent = JSON.stringify(payload, null, 2);

    if (!response.ok) {
      requestHint.textContent = payload.detail || "检测失败，请稍后重试。";
      return;
    }

    predictedLabel.textContent = payload.predicted_label;
    confidenceText.textContent = `置信度：${(payload.confidence * 100).toFixed(2)}%`;
    elapsedMs.textContent = `${payload.elapsed_ms.toFixed(3)}`;
    imageDimensions.textContent = `${payload.original_image_size.width} × ${payload.original_image_size.height}`;

    const okProbability = Number(payload.probabilities["合格件"] || 0);
    const defectProbability = Number(payload.probabilities["缺陷件"] || 0);
    probOkValue.textContent = `${(okProbability * 100).toFixed(2)}%`;
    probDefectValue.textContent = `${(defectProbability * 100).toFixed(2)}%`;
    probOkBar.style.width = `${(okProbability * 100).toFixed(2)}%`;
    probDefectBar.style.width = `${(defectProbability * 100).toFixed(2)}%`;

    requestHint.textContent = `检测完成，请求编号：${payload.request_id}`;
  } catch (error) {
    jsonOutput.textContent = JSON.stringify({ error: String(error) }, null, 2);
    requestHint.textContent = "接口调用失败，请检查服务状态。";
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "开始检测";
  }
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const [file] = event.dataTransfer.files;
  setFile(file);
});

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setFile(file);
});

predictButton.addEventListener("click", predictImage);
resetButton.addEventListener("click", clearSelection);
copyJsonButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(jsonOutput.textContent);
    copyJsonButton.textContent = "已复制";
    setTimeout(() => {
      copyJsonButton.textContent = "复制 JSON";
    }, 1200);
  } catch (error) {
    copyJsonButton.textContent = "复制失败";
    setTimeout(() => {
      copyJsonButton.textContent = "复制 JSON";
    }, 1200);
  }
});

fetchHealth();
resetResults();
