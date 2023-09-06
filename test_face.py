from sklearn import metrics # it needs to be imported first in Jetson Nano
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import *
from scenario import *
from models import *
from method import *
from scenario.evaluators import *

from utils import seedEverything, create_if_not_exists, AverageMeter


def main(args):
    # seed
    seedEverything(args.seed)
    
    CASIAWEBFACE = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    train_transform = CASIAWEBFACE
    # Train & Test dataset and scenario
    casia_dataset = CASIAWeb15Dataset(root="./data/CASIA-15/", 
                        transform=train_transform)
    lfw_test_dataset = LFWPairDataset(root="./data/",
                        transform=None,
                        data_annot="./data/")
    
    # scenario
    train_scenario = ClassIncremental(
        dataset=casia_dataset, n_tasks=casia_dataset._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=0
    )

    # Verification Scenario: n_tasks arguments 필요 없음.
    # LFW Test Dataset
    lfw_test_scenario = VerificationScenario(
        dataset=lfw_test_dataset, n_tasks=lfw_test_dataset._DEFAULT_N_TASKS, batch_size=args.batch_size, n_workers=0
    )   
    
    # 
    n_classes = casia_dataset.n_classes()
    args.N_CLASSES_PER_TASK = casia_dataset._N_CLASSES_PER_TASK
    args._DEFAULT_N_TASKS = casia_dataset._DEFAULT_N_TASKS
    
    # model
    net = resnet18(n_classes)
    loss = torch.nn.CrossEntropyLoss()
    method = LwF(net, loss, args, None)    

    # evaluator
    lfw_rep_evaluator = VerificationEvaluator(method=method, eval_scenario=lfw_test_scenario, name="Verification")
    test_evaluators = [lfw_rep_evaluator]


    # TODO: logger (wandb & local & tensorboard)
    # def set_loggers()
    #     raise NotImplementedError
    
    # save path
    if args.save_path is not None:
        args.save_path = os.path.join(args.save_path, method.NAME)
        create_if_not_exists(args.save_path)
    
    metrics = []
    # train
    method.train()
    method.net.to(method.device)
    for task_id, train_loader in enumerate(train_scenario):
        #- Start Epoch
        scheduler = None
        # pbar = tqdm(range(args.n_epochs))
        for epoch in range(args.n_epochs):
            #-- Start Iteration
            losses = AverageMeter()
            pbar = tqdm(train_loader)
            for idx, (inputs, labels, task, not_aug_inputs) in enumerate(pbar):
                if args.debug and idx > 5:
                    break
                inputs, labels, not_aug_inputs = inputs.to(method.device), labels.to(method.device), not_aug_inputs.to(method.device)

                if hasattr(method, 'meta_observe'):
                    loss = method.meta_observe(inputs, labels, not_aug_inputs)
                else:
                    loss = method.observe(inputs, labels, not_aug_inputs)
                losses.update(loss, inputs.size(0))
                
                pbar.set_description("[Task:{}|Epoch:{}] Avg Loss: {:.5}".format(task_id+1, epoch+1, losses.avg))
            
            if scheduler is not None:
                scheduler.step()
            
            # pbar.set_description(f"[Task:{task_id+1}|Epoch:{epoch+1}] Avg Loss: {losses.avg:.5}")
            # pbar.set_description("[Task:{}|Epoch:{}] Avg Loss: {:.5}".format(task_id+1, epoch+1, losses.avg))

        
        #- Start Evaluation
        accs = test_evaluators[0].fit(current_task_id=task_id, logger=None)
        metrics.append(accs)
    
        if args.save_path is not None:
            if args.save_model:
                fname = os.path.join(args.save_path, "{}_{}.pth".format(method.NAME, task_id+1))
                torch.save(method.net.state_dict(), fname)
    
    if args.save_path is not None:           
        # save the metrics
        fname = os.path.join(args.save_path, "{}.pkl".format(method.NAME))
        with open(file=fname, mode='wb') as f:
            pickle.dump({'ACC': np.array(metrics)}, f)
                
    
if __name__ == "__main__":
    class Config:
        seed = 0
        batch_size = 512
        lr = 0.01
        nowand = 1
        n_epochs = 20
        debug = 0
        save_path = 'save/casia_15'
        save_model = 1

    args = Config
    main(args)