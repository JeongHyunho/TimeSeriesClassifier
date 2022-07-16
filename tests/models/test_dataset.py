from data.eit_emg_dataset import EitEmgGaitDetection
from data.eit_emg_phase_dataset import EitEmgGaitPhaseDataset
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


def test_eit_emg_dataset():
    train_set = EitEmgGaitDetection()
    val_set = EitEmgGaitDetection(validation=True)
    test_set = EitEmgGaitDetection(test=True)
    print(f'Dataset size: {len(train_set)}(train), {len(val_set)}(val), {len(test_set)}(test)')

    sample, label = train_set[0]
    print(f'Sample: {sample}, {label}')


def test_gait_phase_dataset():
    train_set = EitEmgGaitPhaseDataset()
    val_set = EitEmgGaitPhaseDataset(validation=True)
    test_set = EitEmgGaitPhaseDataset(test=True)
    print(f'Dataset size: {len(train_set)}(train), {len(val_set)}(val), {len(test_set)}(test)')

    sample, label = train_set[0]
    print(f'Sample: {sample}, {label}')
