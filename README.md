# TFLite - Project

In this project we test 10 lightweight models on their ability to be deployed on MCUs (Memory Limitation of 256KB).

To achieve this we use full quantization (1-byte integers).

At the same time, we get metrics for multiply-accumulate operations (MACs) performed by the models in order to calculate each model's latency.