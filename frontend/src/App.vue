<script setup lang="ts">
import { ref } from 'vue'

interface PredictionResult {
  filename: string
  anomaly_score: number
  is_anomaly: boolean
  threshold: number
  n_windows: number
  diagnosis: string
}

const API_URL = 'http://localhost:8000'

const selectedFile = ref<File | null>(null)
const result = ref<PredictionResult | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  selectedFile.value = target.files?.[0] ?? null
  result.value = null
  error.value = null
}

async function analyze() {
  if (!selectedFile.value) return

  loading.value = true
  error.value = null
  result.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail ?? `Server error (${response.status})`)
    }

    result.value = await response.json()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Unknown error'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="container">
    <h1> Acoustic Anomaly Detector</h1>
    <p class="subtitle">Upload a machine sound (.wav) to check for anomalies.</p>

    <div class="upload-box">
      <input type="file" accept=".wav" @change="onFileChange" />
      <button :disabled="!selectedFile || loading" @click="analyze">
        {{ loading ? 'Analyzing…' : 'Analyze' }}
      </button>
    </div>

    <p v-if="error" class="error">⚠ {{ error }}</p>

    <div v-if="result" class="result" :class="{ anomaly: result.is_anomaly }">
      <h2>{{ result.is_anomaly ? '⚠ Anomaly detected' : '✓ Normal' }}</h2>
      <p class="diagnosis">{{ result.diagnosis }}</p>
      <ul class="details">
        <li><span>File</span><span>{{ result.filename }}</span></li>
        <li><span>Score</span><span>{{ result.anomaly_score.toFixed(6) }}</span></li>
        <li><span>Threshold</span><span>{{ result.threshold.toFixed(6) }}</span></li>
        <li><span>Windows</span><span>{{ result.n_windows }}</span></li>
      </ul>
    </div>
  </main>
</template>

<style scoped>
.container {
  max-width: 540px;
  margin: 3rem auto;
  font-family: system-ui, sans-serif;
  padding: 0 1rem;
}
h1 { font-size: 1.6rem; }
.subtitle { color: #666; margin-bottom: 2rem; }
.upload-box {
  display: flex; gap: 1rem; align-items: center;
  padding: 1.5rem; border: 2px dashed #ccc; border-radius: 8px;
}
button {
  padding: 0.6rem 1.2rem; border: none; border-radius: 6px;
  background: #2563eb; color: white; cursor: pointer; font-size: 1rem;
}
button:disabled { background: #aaa; cursor: not-allowed; }
.error { color: #c0392b; margin-top: 1rem; }
.result {
  margin-top: 2rem; padding: 1.5rem; border-radius: 8px;
  background: #f0f9f0; border-left: 5px solid #27ae60;
}
.result.anomaly { background: #fdf0f0; border-left-color: #c0392b; }
.diagnosis { font-style: italic; margin: 0.5rem 0 1rem; }
.details { list-style: none; padding: 0; }
.details li {
  display: flex; justify-content: space-between;
  padding: 0.3rem 0; border-bottom: 1px solid #eee;
}
.details span:first-child { color: #888; }
</style>