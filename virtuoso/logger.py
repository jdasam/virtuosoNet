from torch.utils.tensorboard import SummaryWriter
from .loss import get_mean_of_loss_dict

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, loss_dict, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.total_loss", reduced_loss, iteration)
            for key in loss_dict.keys():
                self.add_scalar("traning.{}_loss".format(key), loss_dict[key], iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
    
    def log_multi_perf(self, reduced_loss, loss_dict, grad_norm, iteration):
            self.add_scalar("training.multi_perf.total_loss", reduced_loss, iteration)
            for key in loss_dict.keys():
                self.add_scalar("traning.multi_perf.{}_loss".format(key), loss_dict[key], iteration)
            self.add_scalar("grad.norm.multi_perf", grad_norm, iteration)
    
    def log_style_analysis(self, abs_confusion_matrix, abs_accuracy, norm_confusion_matrix, norm_accuracy, iteration):
        self.add_scalar("validation.style_analysis.abs.accuracy", abs_accuracy, iteration)
        self.add_image("validation.style_analysis.abs.confusion_matrix", abs_confusion_matrix.reshape(1,5,5), iteration)
        self.add_scalar("validation.style_analysis.norm.accuracy", norm_accuracy, iteration)
        self.add_image("validation.style_analysis.norm.confusion_matrix", norm_confusion_matrix.reshape(1,5,5), iteration) 


    def log_validation(self, reduced_loss, loss_dict, model, iteration):
        self.add_scalar("validation.total_loss", reduced_loss, iteration)
        loss_dict = get_mean_of_loss_dict(loss_dict)
        for key in loss_dict.keys():
            self.add_scalar("validation.{}_loss".format(key), loss_dict[key], iteration)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
