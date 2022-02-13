from data.snuh_dataset import SnuhGaitPhase, SnuhEmgForAngle, SnuhEmgForAngleTestStream


def test_snu_gait_dataset():
    dataset = SnuhGaitPhase(split_type='trial')
    sample, label = dataset[0]

    print(f'Dataset size: {len(dataset)}.')
    print(f'Sample : {sample}, {label}.')

    pass


def test_snu_emg_dataset():
    # dataset = SnuhEmgForAngle(motion='gait', target='ankle')
    dataset = SnuhEmgForAngle(motion='gait', target='knee')
    sample, label = dataset[0]

    print(f'Dataset size: {len(dataset)}.')
    print(f'Sample : {sample}, {label}.')

    pass


def test_snu_emg_dataset_test_stream():
    dataset = SnuhEmgForAngleTestStream(motion='gait', target='ankle')
    X, Y, HS, TO, id = dataset[0]
    pass
