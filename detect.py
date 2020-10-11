from time import time

import click

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.experimental.tensorrt import Converter, ConversionParams
from tensorflow.python.saved_model import tag_constants

import logging 
logger = tf.get_logger()
logger.setLevel(logging.ERROR) 

def build_tf_native_inferentor(model):
    return model.predict

def build_trt_inferentor(model):
    signature_keys = list(model.signatures.keys())
    infer = model.signatures['serving_default']
    return infer

@click.command('benchmark')
@click.argument('model-path')
@click.option('--batch-size', default=1)
@click.option('--times', default=10)
@click.option('--csv', default='')
@click.option('--heat-up', default=100)
@click.option('--trt', default=False, is_flag=True)
def benchmark(model_path, batch_size, times, csv,heat_up, trt):
    try:
        if trt:
            model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
            infer = build_trt_inferentor(model)
        else:
            model = tf.keras.models.load_model(model_path)
            infer = build_tf_native_inferentor(model)

        images = np.array([np.zeros((224, 224, 3)) for _ in range(0, batch_size)])
        images = tf.constant(images, dtype=tf.float32)
        print(f'batch size: {batch_size}')

        durations = []
        for i in range(0, times + heat_up):
            if i < heat_up:
                print('Heating up ...', end='\r')
                infer(images)
                continue
            before = time()
            infer(images)
            duration = time() - before
            print(f"Inference: {1/duration} FPS   ", end='\r')
            durations.append(duration)

        durations_df = pd.Series(durations)
        print(durations_df.describe())
        if csv:
            durations_df.to_csv(csv)
    finally:
        tf.keras.backend.clear_session()

@click.command('convert')
@click.argument('model-path', type=click.Path(exists=True))
@click.argument('output-path')
@click.option('--precision', default='fp32', type=click.Choice(['fp32', 'fp16', 'int8']))
@click.option('--engine', default=False, is_flag=True)
def convert(model_path, output_path, precision, engine):
    if engine:
        params = ConversionParams(precision_mode=precision.upper(),maximum_cached_engines=16) #large enought?
    else:
        params = ConversionParams(precision_mode=precision.upper())

    converter = Converter(
        input_saved_model_dir=model_path, 
        conversion_params=params
        )
    converter.convert()
    converter.save(output_saved_model_dir=output_path)

@click.command('pylon')
def pylon():
    pass

@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(benchmark)
    cli.add_command(benchmark_trt)
    cli.add_command(convert)
    cli.add_command(pylon)
    cli()