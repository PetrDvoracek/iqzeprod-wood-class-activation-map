from time import time

import click

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

import numpy as np




@click.command('benchmark')
@click.argument('model-path')
@click.option('--batch-size', default=1)
@click.option('--times', default=1000)
def benchmark(model_path, batch_size, times):
    model = tf.keras.models.load_model(model_path)

    images = np.array([np.zeros((224, 224, 3)) for _ in range(0, batch_size)])
    print(f'batch size: {batch_size}')

    for _ in range(0, times):
        before = time()
        model.predict(images)
        print(f'inference time: {time() - before}')

@click.command('benchmark-trt')
@click.argument('model-path')
@click.option('--batch-size', default=1)
@click.option('--times', default=1000)
def benchmark_trt(model_path, batch_size, times):
    model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
    signature_keys = list(model.signatures.keys())
    infer = model.signatures['serving_default']

    images = np.array([np.zeros((224, 224, 3)) for _ in range(0, batch_size)])
    images = tf.constant(images, dtype=tf.float32)
    print(f'batch size: {batch_size}')

    for _ in range(0, times):
        before = time()
        infer(images)
        print(f'inference time: {time() - before}')

@click.command('convert')
@click.argument('model-path', type=click.Path(exists=True))
@click.argument('output-path')
@click.option('--precision', default='fp32', type=click.Choice(['fp32', 'fp16', 'int8']))
@click.option('--engine', default=False, is_flag=True)
def convert(model_path, output_path, precision, engine):
    if precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
    elif precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.int8
    else: 
        precision_mode = trt.TrtPrecisionMode.FP32
    
    if engine:
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                               max_workspace_size_bytes=8000000000)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path,
                                        conversion_params=conversion_params)

    converter.convert()
    converter.save(output_saved_model_dir=output_path)


@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(benchmark)
    cli.add_command(benchmark_trt)
    cli.add_command(convert)
    cli()