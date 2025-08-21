from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    output_size: int = 1  # regressão univariada por padrão


def build_lstm_model(cfg: LSTMConfig, lookback: int) -> tf.keras.Model:
    """Cria um modelo LSTM empilhado para regressão pontual.

    Entrada: (batch, lookback, input_size)
    Saída: (batch, output_size)
    """
    inputs = tf.keras.Input(shape=(lookback, cfg.input_size))

    x = inputs
    for i in range(cfg.num_layers - 1):
        if cfg.bidirectional:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=cfg.hidden_size,
                    return_sequences=True,
                    dropout=cfg.dropout,
                    recurrent_dropout=0.0,
                )
            )(x)
        else:
            x = tf.keras.layers.LSTM(
                units=cfg.hidden_size,
                return_sequences=True,
                dropout=cfg.dropout,
                recurrent_dropout=0.0,
            )(x)

    # Última camada LSTM sem return_sequences
    if cfg.bidirectional:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=cfg.hidden_size,
                return_sequences=False,
                dropout=cfg.dropout,
                recurrent_dropout=0.0,
            )
        )(x)
    else:
        x = tf.keras.layers.LSTM(
            units=cfg.hidden_size,
            return_sequences=False,
            dropout=cfg.dropout,
            recurrent_dropout=0.0,
        )(x)

    outputs = tf.keras.layers.Dense(cfg.output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def compile_regression_model(model: tf.keras.Model, lr: float = 1e-3, loss: str = "mse") -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model
