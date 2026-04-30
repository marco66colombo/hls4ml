from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from implementation_helpers import run_implementation_collection_test
from tensorflow.keras.layers import Dense

import hls4ml

test_root_path = Path(__file__).parent
VITIS_UNIFIED_BOARD = 'kv260'
VITIS_UNIFIED_PART = 'xck26-sfvc784-2LV-c'
VITIS_UNIFIED_AXI_MODE = 'axi_master'


@pytest.mark.parametrize('backend', ['VitisUnified'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_dense(test_case_id, backend, io_type, synthesis_config):
    model = tf.keras.models.Sequential()
    model.add(
        Dense(
            2,
            input_shape=(1,),
            name='Dense',
            use_bias=True,
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros',
        )
    )
    model.compile(optimizer='adam', loss='mse')

    x_input = np.random.rand(100, 1).astype(np.float32)
    keras_prediction = model.predict(x_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'latency'
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
        board=VITIS_UNIFIED_BOARD,
        part=VITIS_UNIFIED_PART,
        clock_period='10ns',
        input_type='float',
        output_type='float',
        axi_mode=VITIS_UNIFIED_AXI_MODE,
    )

    hls_model.compile()
    hls_prediction = hls_model.predict(x_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == 'InputLayer'
    assert list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[0]._name
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]

    run_implementation_collection_test(
        config=synthesis_config,
        hls_model=hls_model,
        test_case_id=test_case_id,
        backend=backend,
        metadata={
            'board': VITIS_UNIFIED_BOARD,
            'part': VITIS_UNIFIED_PART,
            'axi_mode': VITIS_UNIFIED_AXI_MODE,
        },
    )
