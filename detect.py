import click

import tensorflow as tf


@click.command('benchmark')
@click.argument('model-path')
def benchmark(model_path):
    model = tf.keras.models.load_model(model_path)
    print(model.summary())

@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(benchmark)
    cli()