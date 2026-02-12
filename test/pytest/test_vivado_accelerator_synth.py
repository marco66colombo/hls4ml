import pytest
from synthesis_helpers import run_synthesis_test
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml


@pytest.mark.parametrize('backend', ['VivadoAccelerator'])
def test_tiny_keras_vivado_accelerator(backend, synthesis_config):
    model = Sequential()
    model.add(Dense(4, input_shape=(8,), activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    hls_config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<16,6>', backend=backend)

    output_dir = f'hls4mlprj_tiny_keras_{backend.lower()}'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        output_dir=output_dir,
        project_name=output_dir,
        backend=backend,
        board='zcu102',
        io_type='io_stream',
        interface='axi_stream',
        hls_config=hls_config,
    )

    baseline_file_name = f'{output_dir}.json'
    run_synthesis_test(
        config=synthesis_config,
        hls_model=hls_model,
        baseline_file_name=baseline_file_name,
        backend=backend,
    )
