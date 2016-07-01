# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nose.tools import assert_equal
from procasl.preprocessing import Average


def test_Average_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(copyfile=True,
    mandatory=True,
    ),
    )
    inputs = Average.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Average_outputs():
    output_map = dict(mean_image=dict(),
    )
    outputs = Average.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
