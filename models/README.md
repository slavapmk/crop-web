# Smoking detector model

`smoking_detector.onnx` is an ONNX export of a single-class YOLO smoking detector from [alihassanml/Smoking-detection-yolo11](https://github.com/alihassanml/Smoking-detection-yolo11). The upstream repository is MIT licensed; the license text is included in `SMOKING_MODEL_LICENSE.txt`.

The detector uses ONNX Runtime DirectML when available, or OpenCV DNN on CPU. Its predictions are approximate and can miss small cigarettes or confuse other objects near the mouth. Tune `smoking.confidence`, `smoking.sample_fps`, and the temporal settings in `config.yaml` against the stream you plan to process.
