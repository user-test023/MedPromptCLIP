import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score
from .build import EVALUATOR_REGISTRY
# from dassl.metrics import compute_accuracy

class EvaluatorBase:
    """Base evaluator."""
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0 

@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""
    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._y_probs = []  
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_probs = []  
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        probs = torch.softmax(mo, dim=1) 
        self._y_probs.extend(probs.cpu().numpy().tolist()) 

        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        # aca = 100.0 * compute_accuracy(self._correct,self._total)
        aca = 100 * self._correct/self._total
        err = 100.0 - aca
        kappa = cohen_kappa_score(self._y_true, self._y_pred, weights=None)#add kappa
        cm = confusion_matrix(self._y_true, self._y_pred)
        cm_norm = cm.astype('float')  / cm.sum(axis=1)[:,  np.newaxis] 
        y_probs = np.array(self._y_probs)   
        specificity_class = [specificity(self._y_true==i, self._y_pred==i) for i in np.unique(self._y_true)]   
        auc_class = [roc_auc_score((self._y_true==i).astype(int), y_probs[:,i]) for i in np.unique(self._y_true)] 
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )
        macro_auc = np.mean(auc_class)  * 100 
        macro_specificity = np.mean(specificity_class)  * 100 
        results = {
        "ACA": aca,
        "Kappa": kappa,
        "Macro_F1": macro_f1,
        "Macro_AUC": macro_auc,
        "Macro_Specificity": macro_specificity,
    }
        
        # Add per-class metrics
        unique_labels = np.unique(self._y_true)
        for i, label in enumerate(unique_labels):
            class_prefix = f"Class_{label}"
            results[f"{class_prefix}_F1"] = 100.0 * f1_score(
                self._y_true == label,
                self._y_pred == label,
                average="binary"
            )
            results[f"{class_prefix}_AUC"] = auc_class[i] * 100
            results[f"{class_prefix}_Specificity"] = specificity_class[i] * 100
        print(
        "=> result\n"
        f"* total: {self._total:,}\n"
        f"* correct: {self._correct:,}\n"        
        f"* accuracy: {aca:.2f}%\n" 
        f"* error: {err:.2f}%\n"       
        f"* Kappa: {kappa:.4f}\n"
        f"* macro_f1: {macro_f1:.2f}%\n"
        f"* auc: {macro_auc:.2f}%\n"
        f"* specificity: {macro_specificity:.2f}%"
        )
        
        # Print per-class metrics if enabled
        if self.cfg.TEST.COMPUTE_CMAT:
            print("\n=> per-class metrics")
            for i, label in enumerate(unique_labels):
                classname = self._lab2cname.get(label, str(label))
                print(
                    f"* class {label} ({classname}): "
                    f"F1={results[f'Class_{label}_F1']:.2f}%, "
                    f"AUC={results[f'Class_{label}_AUC']:.2f}%, "
                    f"Specificity={results[f'Class_{label}_Specificity']:.2f}%"
                )
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results