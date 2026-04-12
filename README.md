# AcousticAnomalyDetector

> ML-powered anomaly detection with explainable AI diagnostics via Claude API.
> 
> ## Overview
> This project detects acoustics anomalies in industrial machinery (fans, pumps, valves) using deep learning in 
> Mel-spectogram features. A FastAPI backend serves predictions and a Claude-powered explanation 
> layer translates model outputs into human-readable diagnostic reports.
> 
> ## Architecture
> [ Upload -> Feature Extraction -> CNN Classifier -> FastAPI -> Claude API -> Vue Frontend
> 
> ## Tech Stack
> | Layer               | Technology                        |
> |---------------------|-----------------------------------|
> | Feature Engineering | LibROSA, Mel-Spektogramm, MFCC    |
> | ML Model            | Tensorflow/Keras -> TFLite Export |
> | API                 | FAST-API, Python                  |
> | LLM                 | Anthropic Claude API              |
> | Frontend            | Vue 3 + TypeScript                |
> 
> ## DataSet
> [MIMII Dataset](https://zenodo.org/record/3384388) - Malfunctioning Industrial Machine Investigation and Inspection
> 
> ## Related Project
> -> [Bearing Fault Detection mit CNN](https://github.com/GaTe92/BearingFaultDetection) - Predecessor Project with 
> CWRU Dataset