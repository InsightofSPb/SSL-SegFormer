import copy
import os
from typing import Dict, Optional, Tuple

from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

class Learner:
    def __init__(self, cfg: Dict, weights_path: Optional[str], logger_name: str, model_id: str = ""):
        self.cfg = cfg
        self.weights_path = weights_path
        self.logger_name = logger_name
        self.model_id = model_id
        self.patience = cfg["train"]["patience"]
        self.task = cfg["model"]["task"]

        self.model = None
        self.trainer = None
        self.data_module = None
        self.test_statistics = {}

    @property
    def monitoring_metric(self) -> str:
        if self.task == "classification":
            return "mIoU"
        elif self.task == "regression":
            return "RMSE"
        else:
            raise NotImplementedError(f"No early stopping metric implemented for {self.task} task!")

    def get_train_data_stats(self, human_data: bool, pseudo_data: bool) -> Dict:
        num_human_pixels, num_human_images = 0, 0
        num_pseudo_pixels, num_pseudo_images = 0, 0
        self.data_module = self.setup_data_module(stage=None, human_data=human_data, pseudo_data=pseudo_data)

        train_loaders = self.data_module.train_dataloader()
        if "human" in train_loaders and human_data:
            num_human_images = len(train_loaders["human"].dataset)
            for batch in train_loaders["human"]:
                num_human_pixels += batch["anno_mask_sum"].sum().item()

        if "pseudo" in train_loaders and pseudo_data:
            num_pseudo_images = len(train_loaders["pseudo"].dataset)
            for batch in train_loaders["pseudo"]:
                num_pseudo_pixels += batch["anno_mask_sum"].sum().item()

        return {
            "human": {"num_images": num_human_images, "num_pixels": num_human_pixels},
            "pseudo": {"num_images": num_pseudo_images, "num_pixels": num_pseudo_pixels},
        }

    @property
    def monitoring_mode(self) -> str:
        if self.task == "classification":
            return "max"
        elif self.task == "regression":
            return "min"
        else:
            raise NotImplementedError(f"No monitoring mode implemented for {self.task} task!")

    def setup_trainer(self, iter_count: int, checkpoint_suffix: str = "human") -> Trainer:
        early_stopping = EarlyStopping(
            monitor=f"Validation/{self.monitoring_metric}",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode=self.monitoring_mode,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_saver = ModelCheckpoint(
            monitor=f"Validation/{self.monitoring_metric}",
            filename=f"{self.cfg['experiment']['id']}_model_{self.model_id}_iter{iter_count}_best_{checkpoint_suffix}",
            mode=self.monitoring_mode,
            save_last=True,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            os.path.join("experiments", self.cfg["experiment"]["id"], checkpoint_suffix),
            name=f"{self.cfg['model']['name']}_{self.model_id}",
            version=iter_count,
            default_hp_metric=False,
        )

        trainer = Trainer(
            accelerator="gpu" if self.cfg["train"]["n_gpus"] > 0 else "cpu",
            devices=self.cfg["train"]["n_gpus"],
            logger=tb_logger,
            max_epochs=self.cfg["train"]["max_epoch"],
            callbacks=[lr_monitor, checkpoint_saver, early_stopping],
            log_every_n_steps=1,
        )

        return trainer

    def setup_data_module(
        self, stage: str = None, human_data: bool = True, pseudo_data: bool = True
    ) -> LightningDataModule:
        data_module = get_data_module(self.cfg, human_data=human_data, pseudo_data=pseudo_data)
        data_module.setup(stage)

        return data_module

    def setup_model(self, iter_count: int = 0, num_train_data: int = 1, checkpoint_path: str = None) -> LightningModule:
        raise NotImplementedError("Learner does not implement 'setup_model()' function!")

    def train(
        self, iter_count: int, checkpoint_path: str, human_data: bool, pseudo_data: bool
    ) -> Tuple[LightningModule, str]:
        raise NotImplementedError("Learner does not implement 'train()' function!")

    def evaluate(self, mission_id: int, human_data: bool, pseudo_data: bool):
        self.data_module = self.setup_data_module(stage=None, human_data=human_data, pseudo_data=pseudo_data)
        self.trainer.test(self.model, self.data_module)
        self.track_test_statistics(mission_id, human_data, pseudo_data)

    def track_classification_metrics(self, num_train_data: int):
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Acc", self.model.test_evaluation_metrics["Test/Acc"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/mIoU", self.model.test_evaluation_metrics["Test/mIoU"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/ECE", self.model.test_evaluation_metrics["Test/ECE"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Precision", self.model.test_evaluation_metrics["Test/Precision"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Recall", self.model.test_evaluation_metrics["Test/Recall"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/F1-Score", self.model.test_evaluation_metrics["Test/F1-Score"], num_train_data
        )

    def track_regression_metrics(self, num_train_data: int):
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/MSE", self.model.test_evaluation_metrics["Test/MSE"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/RMSE", self.model.test_evaluation_metrics["Test/RMSE"], num_train_data
        )

    def track_test_statistics(self, mission_id: int, human_data: bool, pseudo_data: bool):
        train_data_stats = self.get_train_data_stats(human_data, pseudo_data)
        self.test_statistics[mission_id] = {**train_data_stats, **self.model.test_evaluation_metrics.copy()}

        if self.task == "classification":
            self.track_classification_metrics(
                train_data_stats["human"]["num_pixels"] + train_data_stats["pseudo"]["num_pixels"]
            )
        elif self.task == "regression":
            self.track_regression_metrics(
                train_data_stats["human"]["num_pixels"] + train_data_stats["pseudo"]["num_pixels"]
            )


class ModelLearner(Learner):
    def __init__(self, cfg: Dict, weights_path: Optional[str], logger_name: str, model_id: str = ""):
        super(ModelLearner, self).__init__(cfg, weights_path, logger_name, model_id=model_id)

        self.model = self.setup_model(iter_count=0, checkpoint_path=self.weights_path)
        self.trainer = self.setup_trainer(0)
    
    def check_checkpoint_keys(self,checkpoint_path: str):
    # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Check if the checkpoint contains a state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Print the keys in the state dictionary
        print("Checkpoint keys:")

        for key in state_dict.keys():
            print(key)

    def setup_model(self, iter_count: int = 0, num_train_data: int = 1, checkpoint_path: str = None) -> LightningModule:
        model = get_model(
            self.cfg, al_logger_name=self.logger_name, al_iteration=iter_count, num_train_data=num_train_data
        )
        if model is None:
            raise ValueError("get_model вернул None. Проверьте конфигурацию и реализацию get_model.")
        
        checkpoint_data = torch.load(checkpoint_path)
        
        if 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data

        # Adjust state_dict keys if necessary
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        if self.cfg["model"]["num_classes_pretrained"] != self.cfg["model"]["num_classes"]:
                model.replace_output_layer()
        
        # if checkpoint_path:
        #     model = model.load_from_checkpoint(
        #         self.weights_path,
        #         cfg=self.cfg,
        #         al_logger_name=self.logger_name,
        #         al_iteration=iter_count,
        #         num_train_data=num_train_data,
        #     )
        #     if self.cfg["model"]["num_classes_pretrained"] != self.cfg["model"]["num_classes"]:
        #         model.replace_output_layer()

        return model

    def retrain_model(self, iter_count: int, num_train_data: int, checkpoint_path: str):
        self.model = self.setup_model(
            iter_count=iter_count, num_train_data=num_train_data, checkpoint_path=checkpoint_path
        )
        self.trainer.fit(self.model, self.data_module)
        
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            # Load the best model weights using the same method as in setup_model
            checkpoint_data = torch.load(best_model_path)
            
            if 'state_dict' in checkpoint_data:
                state_dict = checkpoint_data['state_dict']
            else:
                state_dict = checkpoint_data

            # Adjust state_dict keys if necessary
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded best model weights from {best_model_path}")
            
            # Log the performance of the best model
            best_score = self.trainer.checkpoint_callback.best_model_score
            print(f"Best model score: {best_score}")
        else:
            print("Warning: Best model not found. Using last weights.")

    @property
    def cfg_fine_tuned(self) -> Dict:
        cft_fine_tuned = copy.deepcopy(self.cfg)
        cft_fine_tuned["model"]["num_classes_pretrained"] = cft_fine_tuned["model"]["num_classes"]
        return cft_fine_tuned

    def train(
        self, iter_count: int, checkpoint_path: str, human_data: bool, pseudo_data: bool
    ) -> Tuple[LightningModule, str]:
        print(f"START {self.cfg['model']['name']}_{self.model_id} TRAINING FROM CHECKPOINT {checkpoint_path}")
        checkpoint_suffix = "human" if human_data and not pseudo_data else "combined"
        self.trainer = self.setup_trainer(iter_count, checkpoint_suffix=checkpoint_suffix)
        self.data_module = self.setup_data_module(stage=None, human_data=human_data, pseudo_data=pseudo_data)

        train_data_stats = self.get_train_data_stats(human_data, pseudo_data)
        self.retrain_model(
            iter_count,
            train_data_stats["human"]["num_images"] + train_data_stats["pseudo"]["num_images"],
            checkpoint_path,
        )

        return self.model, self.trainer.checkpoint_callback.best_model_path