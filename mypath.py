"""
Root path
"""

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapesevent':
            return 'dataset/Cityscapes'
        elif dataset == 'dadaevent':
            return 'dataset/DADA_seg'
        elif dataset == 'apolloscapeevent':
            return 'dataset/seg_mini'
        elif dataset == 'kittievent':
            return 'dataset/KITTI-360_mini/'
        elif dataset == 'bdd':
            return 'dataset/seg_event/'
        elif dataset == 'merge3':
            return ['dataset/Cityscapes',
                    'dataset/KITTI-360_mini/',
                    'dataset/seg_event/']
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
