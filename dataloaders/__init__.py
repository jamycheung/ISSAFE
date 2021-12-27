from dataloaders import cityscapes, dada, apolloscape, kitti, bdd, merge3
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'cityscapesevent':
        train_set = cityscapes.CityscapesRGBEvent(args, split='train')
        val_set = cityscapes.CityscapesRGBEvent(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset == 'dadaevent':
        val_set = dada.DADARGBEvent(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = None
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset == 'apolloscapeevent':
        train_set = apolloscape.ApolloscapeRGBEvent(args, split='train')
        val_set = apolloscape.ApolloscapeRGBEvent(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset == 'kittievent':
        train_set = kitti.KITTIRGBEvent(args, split='train')
        val_set = kitti.KITTIRGBEvent(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset == 'bdd':
        train_set = bdd.BDD(args, split='train')
        val_set = bdd.BDD(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset == 'merge3':
        train_set = merge3.Merge3(args, split='train')
        val_set = merge3.Merge3(args, split='val')
        num_class = val_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = None
    else:
        raise NotImplementedError
    return train_loader, val_loader, test_loader, num_class


