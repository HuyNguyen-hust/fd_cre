import os, sys, logging

import torch
from config import Param
from methods.utils import setup_seed

logger = logging.getLogger(__name__)

def run(args):
    setup_seed(args.seed)
    
    if args.mode == 'ce':
        output_dir = os.path.join('exp', args.dataname, args.mode, str(args.learning_rate), str(args.batch_size))
    elif args.mode == 'fd':
        output_dir = os.path.join('exp', args.dataname, args.mode, str(args.learning_rate), str(args.batch_size), str(args.mu))
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S',
            level = logging.INFO,
            handlers=[logging.FileHandler(os.path.join(output_dir, "log.txt")), logging.StreamHandler(sys.stdout)])
    
    logger.info("hyper-parameter configurations:")
    logger.info(str(args.__dict__))
    
    if args.mode == 'crl':
        from methods.manager import Manager
    else:
        from methods.ce_manager import Manager
    
    manager = Manager(args)
    manager.train(args)


if __name__ == '__main__':
    param = Param() # There are detailed hyper-parameter configurations.
    args = param.args
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.dataname
    args.rel_per_task = 8 if args.dataname == 'FewRel' else 4 
    run(args)
    

   
