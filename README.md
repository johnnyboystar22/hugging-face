# Huggingface BERT Quantization Example

Forked from `https://github.com/huggingface/transformers/tree/v4.9.1`

Original README [here](README_org.md)

Mainly modified the following files:
 * examples/pytorch/question-answering/QAT-qdqbert/
 * src/transformers/models/qdqbert/

Quantized tensors:
 * embedding weights
 * encoder
   * linear layer inputs and weights
   * matmul inputs
   * residual add inputs

## Setup

Build the docker image:
```
docker build . -f examples/pytorch/question-answering/QAT-qdqbert/Dockerfile -t bert_quantization:latest
```

Run the docker:
```
docker run --gpus all --privileged --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 bert_quantization:latest
```

In the container:
```
cd transformers/examples/pytorch/question-answering/
```

## Quantization Aware Fine-tuning

We recommend to calibrate the pretrained model and finetune with quantization in one pass to avoid overfitting:

```
python3 QAT-qdqbert/run_qat_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir QAT-qdqbert/calib/bert-base-uncased \
  --do_calib \
  --calibrator percentile \
  --percentile 99.99 \
  --fp16
```

```
python3 QAT-qdqbert/run_qat_qa.py \
  --model_name_or_path QAT-qdqbert/calib/bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 4e-5 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir QAT-qdqbert/finetuned_int8/bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --save_steps 0 \
  --fp16
```

## Export to ONNX

This branch builds a container with upstream pytorch to get opset 13. To export the model finetuned above:

```
python3 QAT-qdqbert/run_qat_qa.py \
  --model_name_or_path QAT-qdqbert/finetuned_int8/bert-base-uncased \
  --output_dir QAT-qdqbert/ \
  --save_onnx \
  --per_device_eval_batch_size 1 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad \
  --tokenizer_name bert-base-uncased \
  --fp16
```

Use `--recalibrate-weights` to calibrate the weight ranges according to the quantizer axis. Use `--quant-per-tensor` for per tensor quantization (default is per channel).
Recalibrating will affect the accuracy of the model, but the change should be minimal (< 0.5 F1).

## Benchmark the INT8 ONNX model inference with TensorRT using dummy input

```
trtexec --onnx=QAT-qdqbert/model.onnx --explicitBatch --workspace=16384 --int8 --fp16 --shapes=input_ids:64x128,attention_mask:64x128,token_type_ids:64x128 --verbose
```

## Evaluate the INT8 ONNX model inference with TensorRT

```
python3 QAT-qdqbert/evaluate-hf-trt-qa.py \
  --onnx_model_path=QAT-qdqbert/model.onnx \
  --model_name_or_path QAT-qdqbert/finetuned_int8/bert-base-uncased/ \
  --output_dir QAT-qdqbert/ \
  --per_device_eval_batch_size 64 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --int8 \
  --seed 42
```

## Quantization options

Some useful options to support different implementations and optimizations. These should be specified for both calibration and finetuning.

|argument|description|
|--------|-----------|
|`--quant-per-tensor`| quantize weights with one quantization range per tensor |
|`--fuse-qkv` | use a single range (the max) for quantizing QKV weights and output activations  |
|`--clip-gelu N` | clip the output of GELU to a maximum of N when quantizing (e.g. 10) |
|`--disable-dropout` | disable dropout for consistent activation ranges |


## FP32 Fine-tuning for comparison

Finetune a fp32 precision model with:

```
python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir ./finetuned_fp32/bert-base-uncased \
  --save_steps 0 \
  --do_train \
  --do_eval
```

### Export to ONNX

```
python3 run_qa.py \
  --model_name_or_path ./finetuned_fp32/bert-base-uncased \
  --output_dir ./ \
  --save_onnx \
  --per_device_eval_batch_size 1 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad \
  --tokenizer_name bert-base-uncased
```

## FP16 Fine-tuning for comparison

Finetune a high precision model with:

```
python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir ./finetuned_fp16/bert-base-uncased \
  --save_steps 0 \
  --do_train \
  --do_eval \
  --fp16
```

### Export to ONNX

```
python3 run_qa.py \
  --model_name_or_path ./finetuned_fp16/bert-base-uncased \
  --output_dir ./ \
  --save_onnx \
  --per_device_eval_batch_size 1 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad \
  --tokenizer_name bert-base-uncased \
  --fp16
```