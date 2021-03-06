# AUTO-GENERATED by tools/checkspecs.py
# DO NOT EDIT
from nose.tools import assert_equal
from procasl.preprocessing import ControlTagRealign


def test_ControlTagRealign_inputs():
    input_map = dict(control_scans=dict(),
    correct_tagging=dict(usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(copyfile=True,
    mandatory=True,
    ),
    paths=dict(),
    register_to_mean=dict(usedefault=True,
    ),
    tag_scans=dict(),
    )
    inputs = ControlTagRealign.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ControlTagRealign_outputs():
    output_map = dict(realigned_files=dict(),
    realignment_parameters=dict(),
    )
    outputs = ControlTagRealign.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
