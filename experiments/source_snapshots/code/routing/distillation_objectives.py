"""Explicit soft teacher relations, not hard positive/negative pair labels."""
import torch
from torch.nn import functional as F

def relation_loss(student_image,student_text,teacher_image,teacher_text,temperature):
    assert temperature>0
    assert student_image.shape==teacher_image.shape and student_text.shape==teacher_text.shape
    teacher_image=F.normalize(teacher_image.detach().float(),dim=-1)
    teacher_text=F.normalize(teacher_text.detach().float(),dim=-1)
    teacher_logits=teacher_image@teacher_text.T/temperature
    student_logits=F.normalize(student_image.float(),dim=-1)@F.normalize(student_text.float(),dim=-1).T/temperature
    image_to_text=F.kl_div(student_logits.log_softmax(1),teacher_logits.softmax(1),reduction='batchmean')
    text_to_image=F.kl_div(student_logits.T.log_softmax(1),teacher_logits.T.softmax(1),reduction='batchmean')
    return (image_to_text+text_to_image)*.5
