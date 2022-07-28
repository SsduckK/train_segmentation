class DATA_INFO:
    RAW_DATA_PATH = "/media/cheetah/IntHDD/datasets/city"
    TFRECORD_PATH = "/home/cheetah/lee_ws/tfrecord"
    CLASS = ['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
    IMAGE_SHAPE = (1024, 2048, 3)

    # LABEL = {'person': {'id': 0, 'color': (60, 20, 220)},
    #          'rider': {'id': 1, 'color': (0, 0, 255)},
    #          'car': {'id': 2, 'color': (142, 0, 0)},
    #          'truck': {'id': 3, 'color': (70, 0, 0)},
    #          'bus': {'id': 4, 'color': (100, 60, 0)},
    #          'motorcycle': {'id': 5, 'color': (230, 0, 0)},
    #          'bycycle': {'id': 6, 'color': (32, 11, 119)}}

    LABEL = {0: (60, 20, 220),
             1: (0, 0, 255),
             2: (142, 0, 0),
             3: (70, 0, 0),
             4: (100, 60, 0),
             5: (230, 0, 0),
             6: (32, 11, 119)}
