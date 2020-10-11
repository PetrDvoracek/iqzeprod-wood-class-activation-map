from time import time

import click

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.experimental.tensorrt import Converter, ConversionParams
from tensorflow.python.saved_model import tag_constants

IMAGE_GRAB_TIMEOUT = 5000

def load_model_get_inferentor(model_path, is_trt):
    if is_trt:
        model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
        infer = build_trt_inferentor(model)
    else:
        model = tf.keras.models.load_model(model_path)
        infer = build_tf_native_inferentor(model)
    return infer

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
        infer = load_model_get_inferentor(model_path, trt)

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
        params = ConversionParams(precision_mode=precision.upper(),maximum_cached_engines=16) #large enough?
    else:
        params = ConversionParams(precision_mode=precision.upper())

    converter = Converter(
        input_saved_model_dir=model_path, 
        conversion_params=params
        )
    converter.convert()
    # if engine:
    #     def engine_inputter():
    #         for _ in range(1):
    #             inp1 = np.random.normal(size=(1, 224, 224, 3)).astype(np.float32)
    #             inp2 = np.random.normal(size=(1, 224, 224, 3)).astype(np.float32)
    #             yield inp1, inp2
    #     converter.build(input_fn=engine_inputter)
    converter.save(output_saved_model_dir=output_path)

def create_pypylon_convertor(pylon):
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter

def rgb2gray(rgb):
    gray = np.expand_dims(rgb[:,:,1], axis=3)
    return cv2.merge([gray, gray, gray])

def errorOrOk(val):
  if val == 1:
    return 'ok'
  elif val ==0:
    return 'error'
  else:
    return '!!! bad value {}'.format(val)

@click.command('pylon')
@click.argument('model_path')
@click.option('--trt', default=False, is_flag=True)
def pylon(model_path, trt):
    from pypylon import pylon
    infer = load_model_get_inferentor(model_path, trt)

    print('initializing camera ...')
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    convertor = create_pypylon_convertor(pylon)
    durations = []
    try:
        while camera.IsGrabbing():
            before = time()

            image_grab = camera.RetrieveResult(IMAGE_GRAB_TIMEOUT)
            if image_grab.GrabSucceeded():
                image_converted = convertor.Convert(image_grab)
                image = image_converted.GetArray()
                image = rgb2gray(image)
                # cv2.imshow('Inference', image)
                image = cv2.resize(image, (224, 224))
                image2infer = np.expand_dims(image, axis=0)

                if trt:
                    pred = infer(tf.constant(image2infer, dtype=tf.float32))
                    pred_class = np.array(pred['dense'])
                    pred_cam = np.array(pred['class_activation'])
                else:
                    pred_class, pred_cam = infer(image2infer)
                print(f"class: {pred_class}")
                pred = np.argmax(pred_class)
                cam = pred_cam[:,:,:,pred]

                cam = np.expand_dims(cam, axis=3)
                cam = cam[0]
                cam = cv2.merge([cam, cam, cam])
                cam = ndimage.zoom(cam, (224 / cam.shape[1], 224 / cam.shape[0], 1), order=2)
                #cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_NEAREST)
                cam_normal = np.ndarray(cam.shape)
                cam = cv2.normalize(cam,  cam_normal, 0, 255, cv2.NORM_MINMAX)
                cam[cam < 150] = 0
                print(cam.min(), cam.max())
                # cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                img_w_cam = image
                img_w_cam[:,:,0] = cam[:,:,0]
                img_w_cam[img_w_cam > 255] = 255
                img_w_cam[img_w_cam < 0] = 0
                show_img = cv2.cvtColor(img_w_cam, cv2.COLOR_RGB2BGR)
                show_img = cv2.resize(show_img, (1024, 544))
                cv2.putText(show_img, errorOrOk(pred), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (125,255,0), 1)
                cv2.imshow('Inference', show_img)
                # plt.imshow(img_w_cam, cmap='seismic')


                duration = time() - before
                print(f"{duration * 1000:.2f} ms, {1 / duration:.0f} FPS  ", end="\r")
                durations.append(duration)
                
                if cv2.waitKey(1) == 27:
                    break

            image_grab.Release()
    finally:
        camera.StopGrabbing()
        cv2.destroyAllWindows()



@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(benchmark)
    cli.add_command(convert)
    cli.add_command(pylon)
    cli()